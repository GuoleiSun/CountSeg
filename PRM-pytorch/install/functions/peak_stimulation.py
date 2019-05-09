import torch
import torch.nn.functional as F
from torch.autograd import Function

class PeakStimulation_ori(Function):

    @staticmethod
    def forward(ctx, input, target,return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4

        # peak finding
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices  = F.max_pool2d(
            padded_maps,
            kernel_size = win_size, 
            stride = 1, 
            return_indices = True)
        peak_map = (indices == element_map)

        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)

        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags

class PeakStimulation_ori_gt(Function):

    @staticmethod
    def forward(ctx, input, target, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4

        # peak finding
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices  = F.max_pool2d(
            padded_maps,
            kernel_size = win_size, 
            stride = 1, 
            return_indices = True)
        peak_map = (indices == element_map)
        if peak_filter:
            mask = input >= peak_filter(input)
            #mask = input > 0
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        
        crm=input * peak_map.float()
        crm[peak_map^1]=float("-Inf")
        crm1=input * peak_map.float()
        crm1[peak_map^1]=float("-Inf")
        #crm=input.clone()
        #crm1=input.clone()
        max_all=[]
        element_map1 = torch.arange(0, crm.size()[2]*crm.size()[3]).long().view(1, 1, crm.size()[2], crm.size()[3])
        element_map1 = element_map1.to(input.device)
        #print(torch.max(target).cpu().numpy())
        for i in range(int(torch.max(target).cpu().numpy())):
            max_one,indices=F.max_pool2d(crm1,kernel_size =input.size()[2:],return_indices = True)
            max_all.append(max_one)
            crm1[indices == element_map1]=float("-Inf")
        max_val=float("Inf")*torch.ones((batch_size,num_channels)).cuda()
        for i in range(batch_size):
            for j in range(num_channels):
                if target[i,j]>0:
                    max_val[i,j]=max_all[int(target[i,j].cpu().numpy())-1][i,j]
        #max_val=max_val.unsqueeze(2).expand(batch_size, num_channels, crm.size()[2])
        #max_val=max_val.unsqueeze(2).expand(batch_size, num_channels, crm.size()[2],crm.size()[3])
        #print(max_val)
        max_val=max_val.view(batch_size,num_channels,1,1)
        # print(max_val.size())
        mask1=crm>=max_val
        mask1=(mask1&peak_map)
        mask2=peak_map-mask1
        #print('peak sti')
        # for i in range(batch_size):
        #     for j in range(num_channels):
        #         if target[i,j] != torch.sum(mask1[i,j]).float():
        #             print(i,j,target[i,j],torch.sum(mask1[i,j]),torch.sum(peak_map[i,j]))
                    #print(max_val[i,j])
                    #print(torch.sum(mask1[i,j]))
                    #print(crm[i,j])
                    #print(mask1[i,j])
                    #dd
                    #print(crm[i,j])
                    #print(max_val[i,j])
        #print('peak end')
        # peak aggregation
        # peak_map2=peak_map.clone()
        if return_aggregation:
            peak_map = peak_map.float()
            mask1=mask1.float()
            mask2=mask2.float()
            #ctx.save_for_backward(input, peak_map)
            ctx.save_for_backward(input, peak_map)
            # return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
            #     peak_map.view(batch_size, num_channels, -1).sum(2), mask1

            ## for selecting images for flow-graph
            # input2=input.clone()
            # input2[peak_map2^1]=float('-Inf')
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                peak_map.view(batch_size, num_channels, -1).sum(2), mask1
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output1,grad_output2):
        input, peak_map = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        # grad_input = mask1 * grad_output1.view(batch_size, num_channels, 1, 1)+mask2*grad_output2.view(batch_size, num_channels, 1, 1)
        grad_input = peak_map * grad_output1.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags

def peak_stimulation_ori(input, target ,return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_ori.apply(input, target,return_aggregation, win_size, peak_filter)

def peak_stimulation_ori_gt(input,target, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_ori_gt.apply(input, target,return_aggregation, win_size, peak_filter)
