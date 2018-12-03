import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from nest import register
import torch
import math

class TableModule(torch.nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()
        
    def forward(self, x):
        y = x.chunk(2, 1)
        return y

class TableModule2(torch.nn.Module):
    def __init__(self):
        super(TableModule2, self).__init__()
        
    def forward(self, x):
        y = x.chunk(3, 1)
        return y

@register
def class_reg_loss(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    weight: float = 0.1) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    # gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    #input1=input[:,0:num_class]
    #input2=input[:,num_class:]
    #input=torch.chunk(input,2,dim=1)
    input=TableModule()(input)
    #print(input1.size())
    #print(target.size())
    #print(loss2(input[1][index2], gt[index2]))
    #print(F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    #SSSprint(0.5*loss2(input[1][index2], gt[index2])+F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    return weight*loss2(input[1][index2], gt[index2])+(1-weight)*F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce)
    #return 0.5*loss2(input2[index2], gt[index2])+torch.nn.CrossEntropyLoss()(input1, target.squeeze().long())

@register
def class_reg_loss_nw(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    weight: float = 0.1) -> Tensor:
    """ no weight.
    """
    gt=target.clone()

    index2=gt!=0
    target[index2]=1
    index2_2=gt<=4
    index2_4=gt>=5
    index2=index2&index2_2
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    loss5 = torch.nn.MarginRankingLoss(margin=0.0)

    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)

    loss_all=loss2(aggregation1[index2], gt[index2])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

    return loss_all
    #return 0.5*loss2(input2[index2], gt[index2])+torch.nn.CrossEntropyLoss()(input1, target.squeeze().long())


@register
def class_reg_loss_all(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    weight: float = 0.1) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    # gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    #input1=input[:,0:num_class]
    #input2=input[:,num_class:]
    #input=torch.chunk(input,2,dim=1)
    input=TableModule()(input)
    #print(input1.size())
    #print(target.size())
    #print(loss2(input[1][index2], gt[index2]))
    #print(F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    #SSSprint(0.5*loss2(input[1][index2], gt[index2])+F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    return weight*loss2(input[1], gt)+(1-weight)*F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce)
    #return 0.5*loss2(input2[index2], gt[index2])+torch.nn.CrossEntropyLoss()(input1, target.squeeze().long())


@register
def class_reg_loss97(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    weight: float = 0.1) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    index2_2=gt<=4
    index2=index2&index2_2
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    #input1=input[:,0:num_class]
    #input2=input[:,num_class:]
    #input=torch.chunk(input,2,dim=1)
    input=TableModule()(input)
    #print(input1.size())
    #print(target.size())
    #print(loss2(input[1][index2], gt[index2]))
    #print(F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    #SSSprint(0.5*loss2(input[1][index2], gt[index2])+F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    return weight*loss2(input[1][index2], gt[index2])+(1-weight)*F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce)
    #return 0.5*loss2(input2[index2], gt[index2])+torch.nn.CrossEntropyLoss()(input1, target.squeeze().long())

@register
def class_reg_loss98_6(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    weight: float = 0.1) -> Tensor:
    """add margin ranking loss
    """
    gt=target.clone()

    index2=gt!=0
    target[index2]=1
    index2_2=gt<=4
    index2_4=gt>=5
    index2=index2&index2_2
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    loss5 = torch.nn.MarginRankingLoss(margin=0.0)

    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)

    loss_all=loss2(aggregation1[index2], gt[index2])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)
    if torch.sum(index2_4)!=0:
        num_ins_5=torch.sum(index2_4)
        loss_all=loss_all+0.1*loss5(aggregation1[index2_4],5*torch.ones((num_ins_5,)).cuda(),torch.ones((num_ins_5,)).cuda())
    return loss_all

@register
def class_reg_loss98_6_2(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    weight: float = 0.1) -> Tensor:
    """add margin ranking loss
    """
    gt=target.clone()

    index2=gt!=0
    target[index2]=1
    index2_2=gt<=4
    index2_4=gt>=5
    index2=index2&index2_2
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    loss5 = torch.nn.MarginRankingLoss(margin=0.0)

    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)

    loss_all=loss2(aggregation1[index2], gt[index2])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)
    if torch.sum(index2_4)!=0:
        num_ins_5=torch.sum(index2_4)
        loss_all=loss_all+0.05*loss5(aggregation1[index2_4],5*torch.ones((num_ins_5,)).cuda(),torch.ones((num_ins_5,)).cuda())
    return loss_all

@register
def class_reg_loss98_6_3(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,    
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    weight: float = 0.1) -> Tensor:
    """add margin ranking loss
    """
    gt=target.clone()

    index2=gt!=0
    target[index2]=1
    index2_2=gt<=4
    index2_4=gt>=5
    index2=index2&index2_2
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    loss5 = torch.nn.MarginRankingLoss(margin=0.0)

    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)

    loss_all=loss2(aggregation1[index2], gt[index2])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)
    if torch.sum(index2_4)!=0:
        num_ins_5=torch.sum(index2_4)
        loss_all=loss_all+0.5*loss5(aggregation1[index2_4],5*torch.ones((num_ins_5,)).cuda(),torch.ones((num_ins_5,)).cuda())
    return loss_all

@register
def class_reg_loss2(
    input: Tensor, 
    target: Tensor,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False,
    weight: float = 0.5) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    num_class=int(gt.size()[1])
    loss2 = torch.nn.SmoothL1Loss()
    #input1=input[:,0:num_class]
    #input2=input[:,num_class:]
    #input=torch.chunk(input,2,dim=1)
    input=TableModule()(input)
    #print(input1.size())
    #print(target.size())
    #print(loss2(input[1][index2], gt[index2]))
    #print(F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    #SSSprint(0.5*loss2(input[1][index2], gt[index2])+F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    return weight*loss2(input[1][index2], gt[index2])+(1-weight)*F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce)
    #return 0.5*loss2(input2[index2], gt[index2])+torch.nn.CrossEntropyLoss()(input1, target.squeeze().long())

@register
def class_reg_loss3(
    input: Tensor, 
    target: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt2=target.clone()
    gt2[:]=0
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    loss3 = torch.nn.MSELoss()
    #input1=input[:,0:num_class]
    #input2=input[:,num_class:]
    #input=torch.chunk(input,2,dim=1)
    input=TableModule2()(input)
    #print(input1.size())
    #print(target.size())
    # print('1:',loss2(input[1], gt))
    # print('2:',loss3(input[2], gt2))
    # print('3:',F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    #print(F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    #SSSprint(0.5*loss2(input[1][index2], gt[index2])+F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    return weight*(loss2(input[1], gt)+loss3(input[2], gt2))+(1-weight)*F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce)
    #return 0.5*loss2(input2[index2], gt[index2])+torch.nn.CrossEntropyLoss()(input1, target.squeeze().long())

def gauss_filter(kernel_size,sigma):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(20, 1, 1, 1)
    gaussian_filter = torch.nn.Conv2d(in_channels=20, out_channels=20,
                                kernel_size=kernel_size,padding=0.5*(kernel_size-1), groups=20, bias=False)
    gaussian_filter.weight.data = gaussian_kernel.cuda()
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

@register
def class_reg_loss4(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt2=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    #input1=input[:,0:num_class]
    #input2=input[:,num_class:]
    #input=torch.chunk(input,2,dim=1)
    gaussian_filter=gauss_filter(5,1)
    #gaussian_filter=gauss_filter(11,3)
    # import matplotlib.pyplot as plt
    
    output4=gaussian_filter(output3)
    output4.detach_()
    return weight*(loss2(output2, 100*output4))+(1-weight)*F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)


@register
def class_reg_loss5(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt2=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    #input1=input[:,0:num_class]
    #input2=input[:,num_class:]
    #input=torch.chunk(input,2,dim=1)
    gaussian_filter=gauss_filter(5,1)
    #gaussian_filter=gauss_filter(11,3)
    import matplotlib.pyplot as plt
    
    output4=gaussian_filter(output3)
    output4.detach_()
    # for i in range(int(target.size()[0])):
    #     for j in range(int(target.size()[1])):
    #         if gt2[i,j].float()>1:
    #             f, axarr = plt.subplots(1,2)
    #             print(gt2[i,j],torch.sum(output3[i,j]),torch.sum(output4[i,j]))
    #             # _,ind=torch.max(gt2)
    #             # print(ind)
    #             axarr[0].imshow(output3[i,j,:,:].detach())
    #             axarr[1].imshow(output4[i,j,:,:])
    #             print(output4[i,j,:,:])
    #             # print(gt2[i,j])
    #             print(torch.sum(output3[i,j,:,:]))
    #             print(torch.sum(output4[i,j,:,:]))
    #             dd
    #print(input1.size())
    #print(target.size())
    # print('1:',loss2(input[1], gt))
    # print('2:',loss3(input[2], gt2))
    # print('3:',F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    #print(F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    #SSSprint(0.5*loss2(input[1][index2], gt[index2])+F.multilabel_soft_margin_loss(input[0], target, None, size_average, reduce))
    return weight*(loss2(output2[index2], 30*output4[index2]))+(1-weight)*F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss6(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt2=target.clone()
    # gt=gt.type(torch.cuda.LongTensor)
    # gt=scatter_(1, gt, 1)
    gt=gt.squeeze()
    index2=gt!=0
    target[index2]=1
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    #input1=input[:,0:num_class]
    #input2=input[:,num_class:]
    #input=torch.chunk(input,2,dim=1)
    gaussian_filter=gauss_filter(5,1)
    #gaussian_filter=gauss_filter(11,3)
    import matplotlib.pyplot as plt
    
    output4=gaussian_filter(output3)
    output4.detach_()
    # for i in range(int(target.size()[0])):
    #     for j in range(int(target.size()[1])):
    #         if gt2[i,j].float()-torch.sum(output4[i,j])>0.5:
    #             f, axarr = plt.subplots(1,2)
    #             print(gt2[i,j],torch.sum(output3[i,j]),torch.sum(output4[i,j]))
    #             # _,ind=torch.max(gt2)
    #             # print(ind)
    #             axarr[0].imshow(output3[i,j,:,:].detach())
    #             axarr[1].imshow(output4[i,j,:,:])
    #             print(output4[i,j,:,:])
    #             # print(gt2[i,j])
    #             print(torch.sum(output3[i,j,:,:]))
    #             print(torch.sum(output4[i,j,:,:]))
    #             dd
    return weight*(loss2(output2, output4))+(1000-weight)*F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss7(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    gaussian_filter=gauss_filter(5,1)
    output4=gaussian_filter(output3)
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()
    # print(index3.size(),index4.size())
    return loss2(output2[index3], output4[index3])+loss3(output2[index4],output4[index4])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss8(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    index1_pre=input>=0
    index1=(index1&index1_pre)
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    gaussian_filter=gauss_filter(5,1)
    output4=gaussian_filter(output3)
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()
    # print(index3.size(),index4.size())
    if torch.sum(index4)!=0:
        return loss2(output2[index3], output4[index3])+loss3(output2[index4],output4[index4])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)
    else:
        return loss2(output2[index3], output4[index3])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss9(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()
    # print(index3.size(),index4.size())
    return loss2(output2[index3], output4[index3])+loss3(output2[index4],output4[index4])+loss4(aggregation1,gt)+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss9_bce1(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()
    # print(index3.size(),index4.size())
    return loss2(output2[index3], output4[index3])+loss4(aggregation1,gt)+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss96(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    index2_2=gt<=4
    index2=index2&index2_2
    target[gt!=0]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()
    # print(index3.size(),index4.size())
    return loss2(output2[index3], output4[index3])+loss3(output2[index4],output4[index4])+loss4(aggregation1[index2_2],gt[index2_2])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss96_7_2(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
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
def class_reg_loss96_7_4(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
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
def class_reg_loss96_7_2_2(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
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
        loss_all=loss_all+0.05*loss5(aggregation1[index2_4],5*torch.ones((num_ins_5,)).cuda(),torch.ones((num_ins_5,)).cuda())
    return loss_all

@register
def class_reg_loss96_7_2_3(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
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
        loss_all=loss_all+0.5*loss5(aggregation1[index2_4],5*torch.ones((num_ins_5,)).cuda(),torch.ones((num_ins_5,)).cuda())
    return loss_all

@register
def class_reg_loss93(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """
    false positive+random negative+non-zero
    """
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()
    index_fp=(index1&(input>0))
    index_neg=(index1&(input<0))

    index_neg3=index_neg.float()*torch.rand((gt.size()[0],gt.size()[1])).cuda().float()
    index_neg3=index_neg3.float()<torch.sum(index_fp).float()/torch.sum(index_neg).float()
    index_neg=index_neg3&index_neg
    index_all=index_fp|index2
    index_all=index_all|index_neg
    # print(torch.sum(index_fp),torch.sum(index_neg))
    if torch.abs(torch.sum(index_fp)-torch.sum(index_neg))>8:
        print(torch.sum(index_fp),torch.sum(index_neg))
        # dd
    # print(index3.size(),index4.size())
    return loss2(output2[index3], output4[index3])+loss3(output2[index4],output4[index4])+loss4(aggregation1[index_all],gt[index_all])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss94(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """
    weight 0.5 for bce loss
    """
    weight=0.5
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()

    return weight*loss2(output2[index3], output4[index3])+weight*loss3(output2[index4],output4[index4])+loss4(aggregation1,gt)+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)

@register
def class_reg_loss95(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """
    weight 0.5 for bce loss
    """
    weight=0.1
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()

    return weight*loss2(output2[index3], output4[index3])+weight*loss3(output2[index4],output4[index4])+loss4(aggregation1,gt)+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)


@register
def class_reg_loss92(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.BCEWithLogitsLoss()
    loss3 = torch.nn.BCEWithLogitsLoss()
    loss4 = torch.nn.MSELoss()
    # gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()
    # print(index3.size(),index4.size())
    return loss2(output2[index3], output4[index3])+loss3(output2[index4],output4[index4])+loss4(aggregation1[index2],gt[index2])+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)


@register
def class_reg_loss10(
    input: Tensor, 
    target: Tensor,
    output2: Tensor,
    output3: Tensor,
    weight: float = 0.5,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Cross entropy loss.
    """
    gt=target.clone()
    gt=gt.squeeze()
    index1=gt==0
    # index1_pre=input>=0
    # index1=(index1&index1_pre)
    index2=gt!=0
    target[index2]=1
    batch_size=int(gt.size()[0])
    num_class=int(gt.size()[1])
    loss2 = torch.nn.MSELoss()
    loss3 = torch.nn.MSELoss()
    loss4 = torch.nn.MSELoss()
    gaussian_filter=gauss_filter(5,1)
    # output4=gaussian_filter(output3)
    output4=output3
    index3=output4!=0
    index3=(index2.view(index2.size()[0],index2.size()[1],1,1)&index3)
    index4=output2!=0
    aggregation1 = F.adaptive_avg_pool2d(output2, 1).squeeze(2).squeeze(2)
    index4=(index1.view(index1.size()[0],index1.size()[1],1,1)&index4)
    output4.detach_()
    # print(torch.sum(output4[index4]))
    # if torch.sum(output4[index4])>0 or torch.sum(output4[index4])<0:
    #     print(torch.sum(output4[index4]))
    #     dd

    # if loss3(output2[index4],0*output4[index4]) != loss3(output2[index4],196*output4[index4]):
    #     print(loss3(output2[index4],0*output4[index4]))
    #     print(loss3(output2[index4],50*output4[index4]))
    #     dd
    # # print(loss3(output2[index4],0*output4[index4]))
    # # print(loss3(output2[index4],196*output4[index4]))
    # # print(index3.size(),index4.size())
    oi=-1.0*output4[index4]
    # # if torch.sum(oi)>0 or torch.sum(oi)<0:
    # #     dd
    # # print(index4.type())
    # print(oi.type())
    # if torch.sum(index4)<=0:
    #     print(index4.size())
    #     dd
    # return loss2(output2[index3], 196*output4[index3])+loss3(output2[index4],oi)+loss4(aggregation1,gt)+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)
    return loss3(output2[index4],oi)+loss4(aggregation1,gt)+F.multilabel_soft_margin_loss(input, target, None, size_average, reduce)


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