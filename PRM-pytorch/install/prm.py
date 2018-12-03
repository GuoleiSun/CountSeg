from typing import Union, Optional, List, Tuple

import cv2
import numpy as np
import torch.nn as nn
from torchvision import models
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import center_of_mass
from nest import register

from .models import FC_ResNet
from .models import FC_VGG16
from .models import FC_VGG16_2
from .modules import PeakResponseMapping
import matplotlib.pyplot as plt



@register
def fc_resnet50(num_classes: int = 20, pretrained: bool = True, selu: bool = False) -> nn.Module:
    """FC ResNet50.
    """
    model = FC_ResNet(models.resnet50(pretrained), num_classes, selu)
    return model

@register
def fc_vgg16(num_classes: int = 20, pretrained: bool = True, selu: bool = False) -> nn.Module:
    """FC ResNet50.
    """
    model = FC_VGG16(models.vgg16(pretrained), num_classes)
    return model

@register
def fc_vgg16_2(num_classes: int = 20, pretrained: bool = True, selu: bool = False) -> nn.Module:
    """FC ResNet50.
    """
    model = FC_VGG16_2(models.vgg16(pretrained), num_classes)
    return model

@register
def peak_response_mapping(
    backbone: nn.Module,
    enable_peak_stimulation: bool = True,
    enable_peak_backprop: bool = True,
    win_size: int = 3,
    sub_pixel_locating_factor: int = 1,
    filter_type: Union[str, int, float] = 'median',
    peak_stimulation: str = 'rel') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping(
        backbone, 
        enable_peak_stimulation = enable_peak_stimulation,
        enable_peak_backprop = enable_peak_backprop, 
        win_size = win_size, 
        sub_pixel_locating_factor = sub_pixel_locating_factor, 
        filter_type = filter_type,
        peak_stimulation = peak_stimulation)
    return model


@register
def prm_visualize(
    instance_list: List[dict], 
    class_names: Optional[List[str]]=None,
    font_scale: Union[int, float] = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction visualization.
    """

    # helper functions
    def rgb2hsv(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        s = 0 if mx == 0 else df / mx
        v = mx
        return h / 360.0, s, v

    def color_palette(N):
        cmap = np.zeros((N, 3))
        for i in range(0, N):
            uid = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (((uid & (1 << 0)) != 0) << 7 - j))
                g = np.bitwise_or(g, (((uid & (1 << 1)) != 0) << 7 - j))
                b = np.bitwise_or(b, (((uid & (1 << 2)) != 0) << 7 - j))
                uid = (uid >> 3)
            cmap[i, 0] = min(r + 86, 255)
            cmap[i, 1] = min(g + 86, 255)
            cmap[i, 2] = b
        cmap = cmap.astype(np.float32) / 255
        return cmap

    if len(instance_list) > 0:
        palette = color_palette(len(instance_list) + 1)
        height, width = instance_list[0]['mask'].shape[0], instance_list[0]['mask'].shape[1]
        instance_mask = np.zeros((height, width, 3), dtype=np.float32)
        peak_response_map = np.zeros((height, width, 3), dtype=np.float32)
        for idx, pred in enumerate(instance_list):
            category, mask, prm = pred['category'], pred['mask'], pred['prm']
            # instance masks
            instance_mask[mask, 0] = palette[idx + 1][0]
            instance_mask[mask, 1] = palette[idx + 1][1]
            instance_mask[mask, 2] = palette[idx + 1][2]
            if class_names is not None:
                y, x = center_of_mass(mask)
                y, x = int(y), int(x)
                text = class_names[category%20]
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                # cv2.putText(
                #     instance_mask,
                #     text,
                #     (x - text_size[0] // 2, y),
                #     font_face,
                #     font_scale,
                #     (1., 1., 1.),
                #     thickness)
            # peak response map
            peak_response = (prm - prm.min()) / (prm.max() - prm.min())
            mask = peak_response > 0.01
            h, s, _ = rgb2hsv(palette[idx + 1][0], palette[idx + 1][1], palette[idx + 1][2])
            peak_response_map[mask, 0] = h
            peak_response_map[mask, 1] = s
            peak_response_map[mask, 2] = np.power(peak_response[mask], 0.5)

        peak_response_map =  hsv_to_rgb(peak_response_map)
        return instance_mask, peak_response_map


@register
def prm_visualize2(
    raw_image: np.ndarray,
    color_base: int,
    instance_list: List[dict], 
    class_names: Optional[List[str]]=None,
    font_scale: Union[int, float] = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction visualization.
    """

    # helper functions
    def rgb2hsv(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        s = 0 if mx == 0 else df / mx
        v = mx
        return h / 360.0, s, v

    def color_palette(N):
        cmap = np.zeros((N, 3))
        for i in range(0, N):
            uid = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (((uid & (1 << 0)) != 0) << 7 - j))
                g = np.bitwise_or(g, (((uid & (1 << 1)) != 0) << 7 - j))
                b = np.bitwise_or(b, (((uid & (1 << 2)) != 0) << 7 - j))
                uid = (uid >> 3)
            cmap[i, 0] = min(r + 86, 255)
            cmap[i, 1] = min(g + 86, 255)
            cmap[i, 2] = b
        cmap = cmap.astype(np.float32) / 255
        return cmap

    if len(instance_list) > 0:
        palette = color_palette(len(instance_list) + 1+100)
        height, width = instance_list[0]['mask'].shape[0], instance_list[0]['mask'].shape[1]
        # instance_mask = np.zeros((height, width, 3), dtype=np.float32)
        peak_response_map = np.zeros((height, width, 3), dtype=np.float32)
        for idx, pred in enumerate(instance_list):
            category, mask, prm = pred['category'], pred['mask'], pred['prm']
            # instance masks
            # instance_mask[mask, 0] = palette[idx + 1][0]
            # instance_mask[mask, 1] = palette[idx + 1][1]
            # instance_mask[mask, 2] = palette[idx + 1][2]
            if class_names is not None:
                y, x = center_of_mass(mask)
                y, x = int(y), int(x)
                text = class_names[category%20]
                font_face = cv2.FONT_HERSHEY_DUPLEX
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                hei=0
                # if idx<=0:
                if idx%2==0:
                    hei=20*idx
                else:
                  hei=30*idx
                # cv2.putText(
                #     raw_image,
                #     text,
                #     (x - text_size[0] // 2, y-hei),
                #     font_face,
                #     font_scale,
                #     (1., 1., 1.),
                #     thickness)
            # peak response map
            peak_response = (prm - prm.min()) / (prm.max() - prm.min())
            # mask = peak_response > 0.01

            # if idx==1:
            #     idx=0
            # elif idx==0:
            #     idx=17

            if idx==6:
                idx=8
            # if idx==2:
            if idx==3:
                idx=2
            elif idx==2:
                idx=3
            h, s, v = rgb2hsv(palette[idx + 1][0], palette[idx + 1][1], palette[idx + 1][2])
            if idx==6:
                s=100
            peak_response_map[mask, 0] = h
            peak_response_map[mask, 1] = s
            peak_response_map[mask, 2] = v

        peak_response_map =  hsv_to_rgb(peak_response_map)
        return raw_image,peak_response_map


@register
def prm_visualize3(
    density: np.ndarray,
    scale: List[float],
    instance_list: List[dict], 
    class_names: Optional[List[str]]=None,
    font_scale: Union[int, float] = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """Prediction visualization.
    """

    # helper functions
    def rgb2hsv(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        s = 0 if mx == 0 else df / mx
        v = mx
        return h / 360.0, s, v

    def color_palette(N):
        cmap = np.zeros((N, 3))
        for i in range(0, N):
            uid = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (((uid & (1 << 0)) != 0) << 7 - j))
                g = np.bitwise_or(g, (((uid & (1 << 1)) != 0) << 7 - j))
                b = np.bitwise_or(b, (((uid & (1 << 2)) != 0) << 7 - j))
                uid = (uid >> 3)
            cmap[i, 0] = min(r + 86, 255)
            cmap[i, 1] = min(g + 86, 255)
            cmap[i, 2] = b
        cmap = cmap.astype(np.float32) / 255
        return cmap

    if len(instance_list) > 0:
        error=0.0
        density2=np.zeros(density.shape+(3,))
        for v in range(len(density2)):
            if v==11:
                mutlipier=2
            else:
                mutlipier=2
            x=mutlipier*density[v,:,:].copy()
            # x=(x-np.min(x))/(np.max(x)-np.min(x))
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(x)
            rgb_img = np.delete(rgba_img, 3, 2)
            density2[v,:,:,:]=rgb_img
        palette = color_palette(len(instance_list) + 1+20)
        height, width = instance_list[0]['mask'].shape[0], instance_list[0]['mask'].shape[1]
        # instance_mask = np.zeros((height, width, 3), dtype=np.float32)
        peak_response_map = np.zeros((height, width, 3), dtype=np.float32)
        for idx, pred in enumerate(instance_list):
            category, mask, prm = pred['category'], pred['mask'], pred['prm']
            # instance masks
            # instance_mask[mask, 0] = palette[idx + 1][0]
            # instance_mask[mask, 1] = palette[idx + 1][1]
            # instance_mask[mask, 2] = palette[idx + 1][2]
            sum_mask=np.sum(density[category][mask])*scale[category]
            error+=np.abs(sum_mask-1)
            # print(mask.shape)
            idx2 = cv2.findContours(mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
            mask2=np.zeros_like(mask)
            # mask2[:]=0
            # print(idx2.shape)
            mask2[idx2[:,0,1],idx2[:,0,0]] = 1
            if class_names is not None:
                y, x = center_of_mass(mask)
                y, x = int(y), int(x)
                text = str(sum_mask)[:3]
                font_face = cv2.FONT_HERSHEY_DUPLEX
                # print("face",font_face)
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                # cv2.putText(
                #     density2[category],
                #     str(class_names[category%20]),
                #     (x - text_size[0] // 2+15, y-text_size[1]+10),
                #     font_face,
                #     font_scale,
                #     (255,255,255),
                #     thickness)

                cv2.putText(
                    density2[category],
                    str(sum_mask)[:3],
                    (x- text_size[0] // 2, y+text_size[1]),
                    font_face,
                    font_scale,
                    (255,255,255),
                    thickness)
                # font                   = cv2.FONT_HERSHEY_SIMPLEX
                # bottomLeftCornerOfText = (10,500)
                # fontScale              = 1
                # fontColor              = (255,255,255)
                # lineType               = 2
                # cv2.putText(img,'Hello World!', 
                #     bottomLeftCornerOfText, 
                #     font, 
                #     fontScale,
                #     fontColor,
                #     lineType)
            # peak response map
            peak_response = (prm - prm.min()) / (prm.max() - prm.min())
            # mask = peak_response > 0.01
            # if idx==3:
            #     idx+=color_base
            h, s, v = rgb2hsv(palette[1][0], palette[1][1], palette[1][2])
            # if idx==3:
            #     s=0
            density2[category,mask2, 0] = 255
            density2[category,mask2, 1] = 255
            density2[category,mask2, 2] = 255

        peak_response_map =  hsv_to_rgb(peak_response_map)
        return density2,peak_response_map,error


@register
def prm_visualize4(
    density: np.ndarray,
    scale: List[float],
    instance_list: List[dict], 
    class_names: Optional[List[str]]=None,
    font_scale: Union[int, float] = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """Prediction visualization.
    """

    # helper functions
    def rgb2hsv(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        s = 0 if mx == 0 else df / mx
        v = mx
        return h / 360.0, s, v

    def color_palette(N):
        cmap = np.zeros((N, 3))
        for i in range(0, N):
            uid = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (((uid & (1 << 0)) != 0) << 7 - j))
                g = np.bitwise_or(g, (((uid & (1 << 1)) != 0) << 7 - j))
                b = np.bitwise_or(b, (((uid & (1 << 2)) != 0) << 7 - j))
                uid = (uid >> 3)
            cmap[i, 0] = min(r + 86, 255)
            cmap[i, 1] = min(g + 86, 255)
            cmap[i, 2] = b
        cmap = cmap.astype(np.float32) / 255
        return cmap

    if len(instance_list) > 0:
        error=0.0
        density2=np.zeros(density[0,:,:].shape+(3,))
        # for v in range(len(density2)):

        #     density2[v,:,:,:]=rgb_img
        palette = color_palette(len(instance_list) + 1+20)
        height, width = instance_list[0]['mask'].shape[0], instance_list[0]['mask'].shape[1]
        # instance_mask = np.zeros((height, width, 3), dtype=np.float32)
        peak_response_map = np.zeros((height, width, 3), dtype=np.float32)
        for idx, pred in enumerate(instance_list):

            category, mask, prm = pred['category'], pred['mask'], pred['prm']
            if category==11:
                mutlipier=2.5
            else:
                mutlipier=2
            x=mutlipier*density[category,:,:].copy()
            # x=(x-np.min(x))/(np.max(x)-np.min(x))
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(x)
            rgb_img = np.delete(rgba_img, 3, 2)
            sum_mask=np.sum(density[category][mask])*scale[category]
            error+=np.abs(sum_mask-1)
            # print(mask.shape)
            idx2 = cv2.findContours(mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
            mask2=np.zeros_like(mask)
            # mask2[:]=0
            # print(idx2.shape)
            mask2[idx2[:,0,1],idx2[:,0,0]] = 1
            if class_names is not None:
                y, x = center_of_mass(mask)
                y, x = int(y), int(x)
                text1 = str(class_names[category%20])
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale=1
                # print("face",font_face)
                thickness = 2
                text_size1, _ = cv2.getTextSize(text1, font_face, font_scale, thickness)
                # cv2.putText(
                #     rgb_img,
                #     text1,
                #     (x - text_size1[0] // 2, y-3),
                #     font_face,
                #     font_scale,
                #     (255,255,255),
                #     thickness)

                text2 = str(sum_mask)[:3]
                text_size2, _ = cv2.getTextSize(text2, font_face, font_scale, thickness)
                cv2.putText(
                    rgb_img,
                    text2,
                    (x- text_size2[0] // 2, y+text_size2[1]-4),
                    font_face,
                    font_scale,
                    (255,255,255),
                    thickness)
                # font                   = cv2.FONT_HERSHEY_SIMPLEX
                # bottomLeftCornerOfText = (10,500)
                # fontScale              = 1
                # fontColor              = (255,255,255)
                # lineType               = 2
                # cv2.putText(img,'Hello World!', 
                #     bottomLeftCornerOfText, 
                #     font, 
                #     fontScale,
                #     fontColor,
                #     lineType)
            # peak response map
            peak_response = (prm - prm.min()) / (prm.max() - prm.min())
            # mask = peak_response > 0.01
            # if idx==3:
            #     idx+=color_base
            h, s, v = rgb2hsv(palette[1][0], palette[1][1], palette[1][2])
            # if idx==3:
            #     s=0
            rgb_img[mask2, 0] = 255
            rgb_img[mask2, 1] = 255
            rgb_img[mask2, 2] = 255
            if idx==0:
                density2=rgb_img
            else:
                density2[mask]=np.maximum(density2[mask],rgb_img[mask])

            text_size1, _ = cv2.getTextSize(text1, font_face, font_scale, thickness)
            
            heig=50
            if category==11:
                heig=40

            # cv2.putText(
            #     density2,
            #     text1,
            #     (x - text_size1[0] // 2, y-heig),
            #     font_face,
            #     font_scale,
            #     (255,255,255),
            #     thickness)


        peak_response_map =  hsv_to_rgb(peak_response_map)
        return density2,peak_response_map,error
