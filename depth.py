import os
import numpy as np
import torch
import torch.nn.functional as F
#import imageio as iio
from PIL import Image
import matplotlib.pyplot as plt

#from plyfile import PlyData, PlyElement
from PIL import Image
from tqdm import tqdm
#from pyquaternion import Quaternion
from nuscenes import NuScenes

#from nuscenes_dataset import NuScenesDataset
#from depth_anything.dpt import DPT_DINOv2, DepthAnything

'''
def get_depth_model(ckpt_path):
    #encoder_type = ['vits', 'vitb', 'vitl']
    
    if 'vitl' in os.path.basename(ckpt_path):
        model = DPT_DINOv2(encoder='vitl')
    elif 'vits' in os.path.basename(ckpt_path):
        model = DPT_DINOv2(encoder='vits', features=64, out_channels=[48,96,192,384])
    elif 'vitb' in os.path.basename(ckpt_path):
        model = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96,192,384,768])
    
    model_sd = torch.load(ckpt_path)
    model.load_state_dict(state_dict=model_sd)
    
    return model


def get_depth(depth_model, rgb, use_cuda=False):
    # depth_model: nn.Module
    # rgb: torch.Tensor (B, 3, H, W) [-1, 1]

    # padding for 14*14 patch
    _, _, h, w = rgb.shape
    mh = h % 14
    mw = w % 14
    if mh != 0: rgb = F.pad(rgb, (0, 0, 0, 14-mh), mode='reflect')
    if mw != 0: rgb = F.pad(rgb, (0, 14-mw, 0, 0), mode='reflect')

    # predict relative depth map
    if use_cuda: rgb = rgb.cuda()
    with torch.no_grad():
        rel_depth = depth_model(rgb)
    #if use_cuda: rel_depth = rel_depth.cpu()
    rel_depth = rel_depth[:, :h, :w].unsqueeze(1) # (B, 1, H, W)

    return rel_depth
'''

'''
def sample_from_depth(depth, h_scale=2, w_scale=2):
    # depth: 1 H W
    hs = h_scale
    ws = w_scale
    _, h, w = depth.shape
    coord = torch.meshgrid(torch.arange(0, h, hs), torch.arange(0, w, ws))
    depth_ = depth[:, 0:h:hs, 0:w:ws] # 1 h w
    coord_ = torch.cat([d.unsqueeze(0) for d in coord], dim=0).type(torch.float32) # 2 h w

    return torch.cat((coord_, depth_), dim=0).flatten(1).permute(1,0) # n 3
'''
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sr = (0, 200) # sample_range

    # load DepthAnything
    #DA = get_depth_model('checkpoints/depth_anything_vitb14.pth')
    #DA.eval().cuda()
    
    # load ZoeDepth
    model_zoe_n = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)

    nusc_pth = '/mnt/jkwang/OccNeRF/data/nuscenes/nuscenes'
    depth_save_pth = '/mnt/jkwang/OccFormer/data/nuscenes_depth'

    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_pth)

    sensor = [
                        'CAM_FRONT', 
                        'CAM_FRONT_RIGHT', 
                        'CAM_BACK_RIGHT', 
                        'CAM_BACK', 
                        'CAM_BACK_LEFT', 
                        'CAM_FRONT_LEFT',
                    ]

    for s in sensor:
        pth = os.path.join(depth_save_pth, 'samples', s)
        if not os.path.exists(pth): os.makedirs(pth)

    for my_scene in tqdm(nusc.scene[sr[0]:sr[1]], position=0, leave=False):

        first_sample_token = my_scene['first_sample_token']
        my_sample = nusc.get('sample', first_sample_token)

        

        sample_count = 0
        while my_sample != '':

            for s in tqdm(sensor, position=1, leave=False):
                data = nusc.get('sample_data', my_sample['data'][s])
                img_pth = os.path.join(nusc.dataroot, data['filename'])
                #img = torch.from_numpy(iio.imread(img_pth).astype(np.float32)).cuda()
                #img = (img.unsqueeze(0).permute(0, 3, 1, 2) / 255) * 2 - 1 # (1, 3, H, W)
                image = Image.open(img_pth).convert("RGB")
                
                with torch.no_grad():
                    depth = zoe.infer_pil(image)

                #with torch.no_grad():
                #    rel_depth = get_depth(DA, img, use_cuda=True) # (6, 1, H, W)
                #    rel_depth = rel_depth.cpu().numpy()

                #np.save(os.path.join(depth_save_pth, data['filename'][:-4] + '_depth.npy'), rel_depth) # (1, 1, H, W)
                depth = np.expand_dims(depth, axis=0)
                depth = np.expand_dims(depth, axis=0)
                np.save(os.path.join(depth_save_pth, data['filename'][:-4] + '_depth.npy'), depth) # (1, 1, H, W)
                

            # next sample
            if my_sample['next'] == '': 
                break
            else:
                my_sample = nusc.get('sample', my_sample['next'])
            sample_count += 1