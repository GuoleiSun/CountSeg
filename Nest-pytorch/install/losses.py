import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from nest import register
import torch
import math

@register
def class_reg_loss98_6(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    tl: int = 5) -> Tensor:
    """
    loss in the first stage for both pascal and coco
    tl: substizing range
    """
    gt=target.clone()

    index2=gt!=0
    target[index2]=1
    index2_2=gt<=tl-1
    index2_4=gt>=tl
    index2=index2&index2_2
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    loss5 = torch.nn.MarginRankingLoss(margin=0.0)

    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)

    loss_all=loss2(aggregation1[index2], gt[index2])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)
    if torch.sum(index2_4)!=0:
        num_ins_5=torch.sum(index2_4)
        loss_all=loss_all+0.1*loss5(aggregation1[index2_4],tl*torch.ones((num_ins_5,)).cuda(),torch.ones((num_ins_5,)).cuda())
    return loss_all

@register
def class_reg_loss96_7_2(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    size_average: bool = True,
    reduce: bool = True) -> Tensor:
    """loss in second stage for pascal
    """
    gt=target.clone()
    target[gt!=0]=1
    p_vs_all=torch.sum(target,0)/target.size()[0]
    p_vs_all=p_vs_all<0.2
    index1=gt==0

    index_neg3=(index1&p_vs_all).float()*torch.rand((gt.size()[0],gt.size()[1])).cuda().float()
    index_neg3=index_neg3.float()<1.1

    index_neg3=index_neg3
    index1_sam=index_neg3&index1
    index1_sam=index1_sam
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    index2_2=gt<=4
    index2_4=gt>=5
    index2=index2&index2_2

    index2_3=index2.clone()
    index2_3=index1_sam|index2_3

    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    loss5 = torch.nn.MarginRankingLoss(margin=0.0)
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    #index4=output2!=0
    index4=(output2!=0)|(output2==0)
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1_sam.view(index1_sam.size()[0],index1_sam.size()[1],1,1)&index4)
    output4.detach_()
    # print(index3.size(),index4.size())
    loss_all=F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)
    if torch.sum(index4)!=0:
        loss_all=loss_all+loss3(output2[index4],output4[index4])
    if torch.sum(index3)!=0:
        loss_all=loss_all+loss2(output2[index3], output4[index3])
    if torch.sum(index2_3)!=0:
        loss_all=loss_all+loss4(aggregation1[index2_3],gt[index2_3])
    if torch.sum(index2_4)!=0:
        num_ins_5=torch.sum(index2_4)
        loss_all=loss_all+0.1*loss5(aggregation1[index2_4],5*torch.ones((num_ins_5,)).cuda(),torch.ones((num_ins_5,)).cuda())
    return loss_all

@register
def class_reg_loss96_7_6(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    tl: float = 0.0,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """loss in second stage for coco
    """
    gt=target.clone()
    target[gt!=0]=1
    p_vs_all=torch.sum(target,0)/target.size()[0]
    p_vs_all=p_vs_all<0.3
    index1=gt==0

    index_neg3=(index1&p_vs_all).float()*torch.rand((gt.size()[0],gt.size()[1])).cuda().float()
    index_neg3=index_neg3.float()<1/10.0

    index_neg3=index_neg3
    index1_sam=index_neg3&index1
    index1_sam=index1_sam
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    index2_2=gt<=tl
    index2_4=gt>=tl+1
    index2=index2&index2_2

    index2_3=index2.clone()
    index2_3=index1_sam|index2_3

    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    loss5 = torch.nn.MarginRankingLoss(margin=0.0)
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    #index4=output2!=0
    index4=(output2!=0)|(output2==0)
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1_sam.view(index1_sam.size()[0],index1_sam.size()[1],1,1)&index4)
    output4.detach_()
    # print(index3.size(),index4.size())
    loss_all=F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)
    if torch.sum(index4)!=0:
        loss_all=loss_all+loss3(output2[index4],output4[index4])
    if torch.sum(index3)!=0:
        loss_all=loss_all+loss2(output2[index3], output4[index3])
    if torch.sum(index2_3)!=0:
        loss_all=loss_all+loss4(aggregation1[index2_3],gt[index2_3])
    if torch.sum(index2_4)!=0:
        num_ins_5=torch.sum(index2_4)
        loss_all=loss_all+0.1*loss5(aggregation1[index2_4],(tl+1)*torch.ones((num_ins_5,)).cuda(),torch.ones((num_ins_5,)).cuda())
    return loss_all

@register
def mse_loss(
    input: Tensor, 
    target: Tensor,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    loss2 = torch.nn.MSELoss()
    # print(gt.size())
    # print(input.size())
    return loss2(input, gt)

@register
def mse_loss2(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    loss2 = torch.nn.MSELoss()
    # print(gt.size())
    # print(input.size())
    return loss2(input, gt)

@register
def mse_loss_cos(
    input: Tensor, 
    target: Tensor,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    index1=gt==0
    index2=gt!=0
    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.MSELoss()
    # print('msecos loss')
    # print(gt.size())
    # print(input.size())
    return (loss1(input[index1], gt[index1])*torch.sum(index1).float()+10*loss2(input[index2], gt[index2])*torch.sum(index2).float())/(10.0*gt.size()[0]*gt.size()[1])


@register
def SmoothL1_loss(
    input: Tensor, 
    target: Tensor,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    loss2 = torch.nn.SmoothL1Loss()
    # print(gt.size())
    # print(input.size())
    return loss2(input, gt)

@register
def cross_entropy_loss(
    input: Tensor, 
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: bool = True,
    ignore_index: int = -100,
    reduce: bool = True) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    # print(gt.size())
    # print(input.size())
    return F.cross_entropy(input, gt, weight, size_average, ignore_index, reduce)


@register
def multilabel_soft_margin_loss(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    weight: Optional[Tensor] = None,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Multilabel soft margin loss.
    """

    if difficult_samples:
        # label 1: positive samples
        # label 0: difficult samples
        # label -1: negative samples
        gt_label = target.clone()
        gt_label[gt_label == 0] = 1
        gt_label[gt_label == -1] = 0
    else:
        gt_label = target
        
    return F.multilabel_soft_margin_loss(input, gt_label, weight, size_average, reduce)

@register
def multilabel_soft_margin_loss2(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    weight: Optional[Tensor] = None,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Multilabel soft margin loss.
    """
    gt=target.clone()
    gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    # gt_label = target
        
    return F.multilabel_soft_margin_loss(input, target, weight, size_average, reduce)