import math

import torch
from nest import register, Context


class AverageMeter(object):
    """Compute the moving average of a variable.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@register
def loss_meter(ctx: Context, train_ctx: Context) -> float:
    """Loss meter.
    """

    if not 'meter' in ctx:
        ctx.meter = AverageMeter()
    # reset at the begining of an epoch
    if train_ctx.batch_idx == 0:
        ctx.meter.reset()
    ctx.meter.update(train_ctx.loss.item(), train_ctx.target.size(0))
    return ctx.meter.avg


@register
def topk_meter(ctx: Context, train_ctx: Context, k: int = 1) -> float:
    """Topk meter.
    """

    # helper function
    def accuracy(output, target, k=1):
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)
    
    if not 'meter' in ctx:
        ctx.meter = AverageMeter()
    # reset at the begining of an epoch
    if train_ctx.batch_idx == 0:
        ctx.meter.reset()
    acc = accuracy(train_ctx.output, train_ctx.target, k)
    ctx.meter.update(acc.item())
    return ctx.meter.avg
