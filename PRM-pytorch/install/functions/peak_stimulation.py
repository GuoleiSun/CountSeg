import torch
import torch.nn.functional as F
from torch.autograd import Function
#from pyinn.ncrelu import ncrelu


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


class PeakStimulation_rel(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
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
        # print(indices.size())
        # print(indices)
        # print(element_map)
        # print(torch.sum(peak_map,(2,3)))

        # peak filtering
        # print(peak_filter(input))
        # print(F.relu(peak_filter(input)))
        if peak_filter:
            mask = input >= peak_filter(input)
            #mask = input > 0
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        # print(peak_map)
        # print(torch.sum(torch.sum(peak_map,3),2))
        # print(torch.sum(peak_map,(2,3)))
        # dd
        # peak aggregation
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                peak_map.view(batch_size, num_channels, -1).sum(2))
            # return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2))
            # peak_map.view(batch_size, num_channels, -1).sum(2))
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        # grad_input = torch.div(peak_map,peak_map.view(batch_size, num_channels, -1).sum(2).view(batch_size,num_channels,1,1)) * grad_output.view(batch_size, num_channels, 1, 1)
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags

class PeakStimulation_rel2(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
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
        # peak aggregation
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                (peak_map.view(batch_size, num_channels, -1).sum(2)+1))
            # return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2))
            # peak_map.view(batch_size, num_channels, -1).sum(2))
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags

class PeakStimulation_rel3(Function):

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
        target_a=target.clone()
        target_a[target_a>0]=1
        target_a=target_a.view(target_a.size()[0],target_a.size()[1],1,1)
        # print(target_a.size())
        # print(peak_filter(input).size())
        if peak_filter:
            mask = input >= torch.mul(peak_filter(input),target_a)
            #mask = input > 0
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        # peak aggregation
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                (peak_map.view(batch_size, num_channels, -1).sum(2)+1))
            # return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2))
            # peak_map.view(batch_size, num_channels, -1).sum(2))
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags


class PeakStimulation_rel3_norelu(Function):

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
        # target_a=target.clone()
        # target_a[target_a>0]=1
        # target_a=target_a.view(target_a.size()[0],target_a.size()[1],1,1)

        if peak_filter:
            # mask = input >= torch.mul(peak_filter(input),target_a)
            mask = input >= peak_filter(input)
            #mask = input > 0
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        # peak aggregation
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                (peak_map.view(batch_size, num_channels, -1).sum(2)+1e-6)
            # return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2))
            # peak_map.view(batch_size, num_channels, -1).sum(2))
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags


class PeakStimulation_ncrelu(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
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
        num_channels=int(num_channels)
        #print(num_channels)
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input[:,0:int(num_channels/2),:,:] * peak_map[:,0:int(num_channels/2),:,:]).view(batch_size, int(num_channels/2), -1).sum(2) / \
                peak_map[:,0:int(num_channels/2),:,:].view(batch_size, int(num_channels/2), -1).sum(2)+(input[:,int(num_channels/2):,:,:] * peak_map[:,int(num_channels/2):,:,:]).view(batch_size, int(num_channels/2), -1).sum(2) / \
                peak_map[:,int(num_channels/2):,:,:].view(batch_size, int(num_channels/2), -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        num_channels=int(num_channels/2)
        grad_input = peak_map * torch.cat([grad_output.view(batch_size, num_channels, 1, 1),grad_output.view(batch_size, num_channels, 1, 1)],1)
        return (grad_input,) + (None,) * ctx.num_flags

class PeakStimulation_class_reg(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
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


class PeakStimulation_sort_boost(Function):

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
        # peak_map1=peak_map.clone()
        # indices1=indices.clone()
        # print(indices.size())
        # print(element_map.size())
        # dd
        # target_a=target.clone()
        # target_a[target_a>0]=1
        # target_a=target_a.view(target_a.size()[0],target_a.size()[1],1,1)
        # print(target_a.size())
        # print(peak_filter(input).size())
        if peak_filter:
            mask = input >= peak_filter(input)
            #mask = input > 0
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        
        crm=input * peak_map.float()
        crm1=input * peak_map.float()
        #crm=input.clone()
        #crm1=input.clone()
        max_all=[]
        element_map1 = torch.arange(0, crm.size()[2]*crm.size()[3]).long().view(1, 1, crm.size()[2], crm.size()[3])
        element_map1 = element_map1.to(input.device)
        #print(torch.max(target).cpu().numpy())
        for i in range(int(torch.max(target).cpu().numpy())):
            max_one,indices=F.max_pool2d(crm1,kernel_size =input.size()[2:],return_indices = True)
            max_all.append(max_one)
            crm1[indices == element_map1]=-float("Inf")
        max_val=10000*torch.ones((batch_size,num_channels)).cuda()
        for i in range(batch_size):
            for j in range(num_channels):
                if target[i,j]>0:
                    max_val[i,j]=max_all[int(target[i,j].cpu().numpy())-1][i,j]
        max_val=max_val.unsqueeze(2).expand(batch_size, num_channels, crm.size()[2])
        max_val=max_val.unsqueeze(2).expand(batch_size, num_channels, crm.size()[2],crm.size()[3])
        # print(max_val.size())

        mask1=crm>=max_val
        mask2=peak_map-mask1
        # print(mask1.view(batch_size, num_channels, -1).sum(2))
        # print(target)
        # print(torch.sum(mask1.view(batch_size, num_channels, -1).sum(2)-target.long()))
        # import matplotlib.pyplot as plt
        # f, axarr = plt.subplots(1,4)
        # print(target.size())
        # _,largest_index=target.max(0)
        # a=2
        # b=11
        # print(indices1[a,b])
        # print(max_val[a,b])
        # print(target[a,:])
        # print(input[a,b,:,:])
        # print(torch.max(input[a,b,0:5,:]))
        # print(torch.max(input[a,b,5:,:]))
        # print(input[a,b,:,:]*mask1[a,b,:,:].float())
        # axarr[0].imshow(input[a,b,:,:])
        # axarr[1].imshow(mask1[a,b,:,:])
        # axarr[2].imshow(input[a,b,:,:]*peak_map[a,b,:,:].float())
        # axarr[3].imshow(input[a,b,:,:]*peak_map1[a,b,:,:].float())
        # print(input[1,largest_index[0,1],:])
        # print(mask1[largest_index[0,0],largest_index[0,1],:])
        # dd
        # peak aggregation
        if return_aggregation:
            peak_map = peak_map.float()
            mask1=mask1.float()
            mask2=mask2.float()
            #ctx.save_for_backward(input, peak_map)
            ctx.save_for_backward(input, mask1, mask2)
            return peak_list, (input*mask1).view(batch_size, num_channels, -1).sum(2), (input*mask2).view(batch_size, num_channels, -1).sum(2)
            # return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
            #     (peak_map.view(batch_size, num_channels, -1).sum(2)+1))
            # return peak_list, F.relu((input * peak_map).view(batch_size, num_channels, -1).sum(2))
            # peak_map.view(batch_size, num_channels, -1).sum(2))
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output1,grad_output2):
        input, mask1,mask2 = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = mask1 * grad_output1.view(batch_size, num_channels, 1, 1)+mask2*grad_output2.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags


class PeakStimulation_ori_gt_flow_graph(Function):

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
        peak_map2=peak_map.clone()
        if return_aggregation:
            peak_map = peak_map.float()
            mask1=mask1.float()
            mask2=mask2.float()
            #ctx.save_for_backward(input, peak_map)
            ctx.save_for_backward(input, peak_map)
            # return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
            #     peak_map.view(batch_size, num_channels, -1).sum(2), mask1

            ## for selecting images for flow-graph
            input2=input.clone()
            input2[peak_map2^1]=float('-Inf')
            return input,input2, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
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

def peak_stimulation_rel(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_rel.apply(input, return_aggregation, win_size, peak_filter)

def peak_stimulation_rel2(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_rel2.apply(input, return_aggregation, win_size, peak_filter)

def peak_stimulation_rel3(input, target, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_rel3.apply(input, target,return_aggregation, win_size, peak_filter)

def peak_stimulation_rel3_norelu(input, target, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_rel3_norelu.apply(input, target,return_aggregation, win_size, peak_filter)

def peak_stimulation_ncrelu(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_ncrelu.apply(input,return_aggregation, win_size, peak_filter)

def peak_stimulation_class_reg(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_class_reg.apply(input, return_aggregation, win_size, peak_filter)

def peak_stimulation_sort_boost(input,target, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_sort_boost.apply(input, target,return_aggregation, win_size, peak_filter)

def peak_stimulation_ori_gt(input,target, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_ori_gt.apply(input, target,return_aggregation, win_size, peak_filter)

def peak_stimulation_ori_gt_selected_flow_graph(input,target, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation_ori_gt_flow_graph.apply(input, target,return_aggregation, win_size, peak_filter)