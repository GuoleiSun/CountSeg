import os
import errno
import logging
from typing import Any, Iterable, Union, List, Tuple, Dict, Callable, Optional

import torch
from torch import Tensor, nn, optim
from torch.utils import data
from tqdm import tqdm, tqdm_notebook
from nest import register, Context
import torch.nn.functional as F
import numpy as np
import gc
import json
from nest import modules


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

def adjust_learning_rate(optimizer, epoch, ori_lr):
    """Sets the learning rate to the initial LR decayed by 2 every 50 epochs"""
    for i,param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = ori_lr[i]*(0.5 ** (epoch // 50))

def save_json(fname, data):
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

@register
def network_trainer(
    data_loaders: Tuple[List[Tuple[str, data.DataLoader]], List[Tuple[str, data.DataLoader]]],
    model: nn.Module,
    criterion1: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    optimizer: Callable[[Iterable], optim.Optimizer],
    criterion2: Optional[Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]] = None,
    parameter: Optional[Callable] = None,
    meters: Optional[Dict[str, Callable[[Context], Any]]] = None,
    hooks: Optional[Dict[str, List[Callable[[Context], None]]]] = None,
    train_method: Optional[str] = 'sep',
    path_history: Optional[str] = './',
    resume: Optional[str] = None,
    log_path: Optional[str] = None,
    max_epoch: int = 200,
    test_interval: int = 1,
    device: str = 'cuda',
    use_data_parallel: bool = True,
    use_cudnn_benchmark: bool = True,
    epoch_stage1: int = 30,
    random_seed: int = 999) -> Context:
    """Network trainer.
    data_loaders: [train set, optional[validation set]]
    criterion1: loss for stage 1
    criterion2: loss for stage 2
    train_method: sep, train stage 1 and stage 2 seprately; e2e, train stage 1 and stage 2 joinly, i.e. in an end to end manner
    epoch_stage1: the number of epoches in the first stage; We use 20 for pascal and 30 for coco
    """

    torch.manual_seed(random_seed)

    # setup training logger
    logger = logging.getLogger('nest.network_trainer')
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    # log to screen
    screen_handler = TqdmHandler()
    screen_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(screen_handler)
    # log to file
    if not log_path is None:
        # create directory first
        try:
            os.makedirs(os.path.dirname(log_path))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        file_handler = logging.FileHandler(log_path, encoding='utf8')
        file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)
    
    # determine which progress bar to use
    def run_in_notebook():
        try:
            return get_ipython().__class__.__name__.startswith('ZMQ')
        except NameError:
            pass
        return False
    progress_bar = tqdm_notebook if run_in_notebook() else tqdm
    
    # setup device
    device = torch.device(device)
    if device.type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA is not available.'
        torch.backends.cudnn.benchmark = use_cudnn_benchmark

    # loaders for train and test splits
    train_loaders, test_loaders = data_loaders
    
    # setup model
    model = model.to(device)

    # multi-gpu support
    if device.type == 'cuda' and use_data_parallel:
        model = nn.DataParallel(model)

    # setup optimizer
    params = model.parameters() if parameter is None else parameter(model)
    optimizer = optimizer(params)

    # resume checkpoint
    start_epoch_idx = 0
    start_batch_idx = 0
    if not resume is None:
        logger.info('loading checkpoint "%s"' % resume)
        checkpoint = torch.load(resume)
        start_epoch_idx = checkpoint['epoch_idx']
        start_batch_idx = checkpoint['batch_idx']
        model_dict = model.state_dict()
        trained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
        model_dict.update(trained_dict)
        model.load_state_dict(model_dict)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info('checkpoint loaded (epoch %d)' % start_epoch_idx)

    # create training context
    ctx = Context(
        split = 'train',
        is_train = True,
        model = model,
        optimizer = optimizer,
        max_epoch = max_epoch,
        epoch_idx = start_epoch_idx,
        batch_idx = start_batch_idx,
        input = Tensor(),
        output = Tensor(),
        output2 = Tensor(),  ##changed
        output3 = Tensor(),  ##changed
        target = Tensor(),
        target1 = Tensor(),
        loss = Tensor(),
        metrics = dict(),
        state_dicts = [],
        eva_metrics='',
        save_dir=path_history,
        is_best = False,
        logger = logger)

    # helper func for executing hooks
    def run_hooks(hook_type):
        if isinstance(hooks, dict) and hook_type in hooks:
            if hook_type=='save_best_model':
                for hook in hooks.get(hook_type):
                    hook(ctx)
            else:
                for hook in hooks.get(hook_type):
                    hook(ctx)                

    # helper func for processing dataset split
    def process(split, data_loader, is_train):
        ctx.max_batch = len(data_loader)
        ctx.split = split
        ctx.is_train = is_train

        run_hooks('on_start_split')

        # set model status
        if is_train:
            model.train() 
            gc.collect()
        else:
            model.eval()

        # iterate over batches
        for batch_idx, (input, target,_) in enumerate(progress_bar(data_loader, ascii=True, desc=split, unit='batch', leave=False)):
            if batch_idx < ctx.batch_idx:
                continue

            # prepare a batch of data
            ctx.batch_idx = batch_idx
            if isinstance(input, (list, tuple)):
                ctx.input = [v.to(device) if torch.is_tensor(v) else v for v in input]
            elif isinstance(input, dict):
                ctx.input = {k: v.to(device) if torch.is_tensor(v) else v for k, v in input.items()}
            else:
                ctx.input = input.to(device)
            ctx.target = target.to(device)

            run_hooks('on_start_batch')

            # compute output and loss
            with torch.set_grad_enabled(is_train):
                ctx.output, ctx.output2, ctx.output3 = ctx.model(ctx.input,ctx.target)
                ctx.loss = criterion(ctx.output, ctx.target, ctx.output2, ctx.output3)

            # measure performance
            if not meters is None:
                ctx.metrics.update({split + '_' + k: v(ctx) for k, v in meters.items() if v is not None})

            # update model parameters
            if is_train:
                optimizer.zero_grad()
                ctx.loss.backward()
                optimizer.step()

            run_hooks('on_end_batch')
            ctx.batch_idx = 0

        run_hooks('on_end_split')

    def mrmse(non_zero,count_pred, count_gt):
        ## compute mrmse
        nzero_mask=torch.ones(count_gt.size())
        if non_zero==1:
            nzero_mask=torch.zeros(count_gt.size())
            nzero_mask[count_gt!=0]=1
        mrmse=torch.pow(count_pred - count_gt, 2)
        mrmse = torch.mul(mrmse, nzero_mask)
        mrmse = torch.sum(mrmse, 0)
        nzero = torch.sum(nzero_mask, 0)
        mrmse = torch.div(mrmse, nzero)
        mrmse = torch.sqrt(mrmse)
    #     print(mrmse.size())
        mrmse = torch.mean(mrmse)
        return mrmse

    def rel_mrmse(non_zero,count_pred, count_gt):
        ## compute relative mrmse
        nzero_mask=torch.ones(count_gt.size())
        if non_zero==1:
            nzero_mask=torch.zeros(count_gt.size())
            nzero_mask[count_gt!=0]=1
        num = torch.pow(count_pred - count_gt, 2)
        denom = count_gt.clone()
        denom = denom+1
        rel_mrmse = torch.div(num, denom)
        rel_mrmse = torch.mul(rel_mrmse, nzero_mask)
        rel_mrmse = torch.sum(rel_mrmse, 0)
        nzero = torch.sum(nzero_mask, 0)
        rel_mrmse = torch.div(rel_mrmse, nzero)
        rel_mrmse = torch.sqrt(rel_mrmse)
        rel_mrmse = torch.mean(rel_mrmse)
        return rel_mrmse


   # training two stages together
    def process2(split, data_loader,is_train,criterion):
        ctx.max_batch = len(data_loader)
        ctx.split = split
        ctx.is_train = is_train

        run_hooks('on_start_split')

        # set model status
        if is_train:
            model.train() 
            gc.collect()
        else:
            model.eval()
            counting_pred=[]
            counting_gt=[]

        # iterate over batches
        for batch_idx, (input, target,target1) in enumerate(progress_bar(data_loader, ascii=True, desc=split, unit='batch', leave=False)):
            if batch_idx < ctx.batch_idx:
                continue

            # prepare a batch of data
            ctx.batch_idx = batch_idx
            if isinstance(input, (list, tuple)):
                ctx.input = [v.to(device) if torch.is_tensor(v) else v for v in input]
            elif isinstance(input, dict):
                ctx.input = {k: v.to(device) if torch.is_tensor(v) else v for k, v in input.items()}
            else:
                ctx.input = input.to(device)
            ctx.target = target.to(device)
            ctx.target1 = target1.to(device)

            run_hooks('on_start_batch')

            # compute output and loss
            with torch.set_grad_enabled(is_train):
                ctx.output, ctx.output2, ctx.output3 = ctx.model(ctx.input,ctx.target)
                if is_train:
                    ctx.loss = criterion(ctx.output, ctx.target, ctx.output2, ctx.output3)

            # measure performance
            if not meters is None:
                ctx.metrics.update({split + '_' + k: v(ctx) for k, v in meters.items() if v is not None})

            # update model parameters if training the model; otherwise, calculate counting prediction
            if is_train:
                optimizer.zero_grad()
                ctx.loss.backward()
                optimizer.step()
            else:
                confidence=ctx.output
                class_response_map1=ctx.output2
                confidence=confidence.cpu().detach().numpy()
                count_one = F.adaptive_avg_pool2d(class_response_map1, 1).squeeze(2).squeeze(2).detach().cpu().numpy()[0]
                confidence[confidence<0]=0
                confidence=confidence[0]
                confidence[confidence>0]=1
                counting_pred.append(np.round(confidence*count_one))
                counting_gt.append(target.detach().cpu().numpy()[0])

            run_hooks('on_end_batch')
            ctx.batch_idx = 0
        if not is_train:
            counting_pred=np.array(counting_pred)
            counting_gt=np.array(counting_gt)
            # print(counting_pred.shape,counting_gt.shape)
            return [mrmse(1,torch.from_numpy(counting_pred).float(), torch.from_numpy(counting_gt).float()),
            rel_mrmse(1,torch.from_numpy(counting_pred).float(), torch.from_numpy(counting_gt).float()),
            mrmse(0,torch.from_numpy(counting_pred).float(), torch.from_numpy(counting_gt).float()),
            rel_mrmse(0,torch.from_numpy(counting_pred).float(), torch.from_numpy(counting_gt).float())]
        else:
            return None

        run_hooks('on_end_split')


    # trainer processing
    run_hooks('on_start')

    ori_lr=[]
    for param_group in optimizer.param_groups:
        ori_lr.append(param_group['lr'])

    history={"best_val_epoch":[-1]*4, "best_val_result":[np.inf]*4}
    eva_metrics_list=['mrmse_nz','rmrmse_nz','mrmse','rmrmse']

    for epoch_idx in progress_bar(range(ctx.epoch_idx, max_epoch), ascii=True, unit='epoch'):
        ctx.epoch_idx = epoch_idx
        run_hooks('on_start_epoch')
        
        adjust_learning_rate(optimizer, epoch_idx, ori_lr)
        for param_group in optimizer.param_groups:
            print('learning rate:', param_group['lr'])

        if train_method=='sep':
            for split, loader in train_loaders:
                process(split, loader, True)
            # testing
            if epoch_idx % test_interval == 0:
                for split, loader in test_loaders:
                    process(split, loader, False)
            run_hooks('on_end_epoch')

        elif train_method=='joint_train':
            assert not(criterion2 is None), "criterion2 not provided"

            ## do training: our training are divided into two stages: first stage and second stage
            if epoch_idx<=epoch_stage1-1:
                print('stage 1 of the training: using criterion1')
                for split, loader in train_loaders:
                    process2(split, loader, True,criterion1)
            else:
                print('stage 2 of the training: using criterion2')
                for split, loader in train_loaders:
                    process2(split, loader, True,criterion2)

            ## do validation
            if len(test_loaders)==0:
                print("no validation during training")
            else:
                print("validation start")
                for split, loader in test_loaders:
                    if epoch_idx<=epoch_stage1-1:
                        results=process2(split, loader, False,criterion1)
                    else:
                        results=process2(split, loader, False,criterion2)
                print("mrmse_nz: %f, rmrmse_nz: %f, mrmse: %f, rmrmse: %f " %(results[0],results[1],
                            results[2],results[3]))

                for i in range(len(results)):
                    if history['best_val_result'][i]>results[i]:
                        history['best_val_epoch'][i]=epoch_idx
                        history['best_val_result'][i]=float(results[i].cpu().numpy())
                        ctx.eva_metrics=eva_metrics_list[i]
                        ctx.is_best=True
            run_hooks('on_end_epoch')
            run_hooks('save_checkpoints')
            save_json(path_history+'/history.json', history)
            print()
            print('--------------------------------------------------------',end='\n\n')

            # if epoch_idx<epoch_stage1-1:
            #     run_hooks('on_end_epoch_save_latest')
            # elif epoch_idx==epoch_stage1-1:
            #     run_hooks('on_end_epoch')
            # else:
            #     if len(test_loaders)==0:
            #         run_hooks('on_end_epoch')
            #     else:
            #         run_hooks('on_end_epoch_save_latest')
            #         print("mrmse: %f, rmrmse: %f, mrmse_nz: %f, rmrmse_nz: %f " %(results[0],results[1],
            #                 results[2],results[3]))
            #         update_flag=0
            #         for i in range(len(results)):
            #             if history['best_val_result'][i]>results[i]:
            #                 history['best_val_epoch'][i]=epoch_idx
            #                 history['best_val_result'][i]=float(results[i].cpu().numpy())
            #                 ctx.eva_metrics=eva_metrics_list[i]
            #                 update_flag=1
            #                 run_hooks('save_best_model')
            #         save_json(path_history, history)

    run_hooks('on_end')

    return ctx
