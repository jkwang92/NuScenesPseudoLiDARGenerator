import os
import numpy as np
import torch
import torch.nn.functional as F
import imageio as iio
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes import NuScenes

#from depth import get_depth_model, get_depth
#from seg import get_sematic_maps, Sem_Predictor, colors
from pc import img2pc, pc2occ, pc_alignment, save_pc_as_ply, get_orb_descriptor


def main(scene_range=(0, 851)):
    # load DepthAnything
    #DA = get_depth_model('checkpoints/depth_anything_vitb14.pth')
    #DA.eval().cuda()

    # load GroundingDINO + SAM
    #SAM = Sem_Predictor(sam_checkpoint='checkpoints/sam_vit_h_4b8939.pth', device='cpu')
    
    nusc_root = '/mnt/jkwang/OccNeRF/data/nuscenes/nuscenes'
    nusc_depth_root = '/mnt/jkwang/OccFormer/data/nuscenes_depth'
    nusc_sem_root = '/mnt/jkwang/OccNeRF/data/nuscenes/nuscenes_semantic'
    
    pcs_root = '/mnt/jkwang/OccFormer/data/nuscenes_occ'

    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root)
    
    # settings
    # voxel
    voxel_range = (-51.2, 51.2)
    voxel_height = (-5.0, 3.0)
    voxel_res = 0.4

    #scene_index = 0
    for scene_index in tqdm(range(*scene_range)):
        my_scene = nusc.scene[scene_index]
        first_sample_token = my_scene['first_sample_token']
        my_sample = nusc.get('sample', first_sample_token)

        #sample_count = 0
        while my_sample != '':
            sensor = [
                        'CAM_FRONT', 
                        'CAM_FRONT_RIGHT', 
                        'CAM_BACK_RIGHT', 
                        'CAM_BACK', 
                        'CAM_BACK_LEFT', 
                        'CAM_FRONT_LEFT',
                    ]
            
            # voxel init
            #xy_num = 2 * (voxel_range[1]-voxel_range[0]) / voxel_res
            #z_num = voxel_height / voxel_res

            #voxel = torch.zeros(xy_num, xy_num, z_num) # x y z

            voxel_coord_x = torch.arange(-voxel_range[1], voxel_range[1]+voxel_res, voxel_res)
            voxel_coord_y = torch.arange(-voxel_range[1], voxel_range[1]+voxel_res, voxel_res) 
            voxel_coord_z = torch.arange(voxel_height[0], voxel_height[1]+voxel_res, voxel_res)
            voxel_coord = torch.meshgrid(voxel_coord_x, voxel_coord_y, voxel_coord_z)
            coord = torch.cat([d.unsqueeze(0).unsqueeze(0) for d in voxel_coord], dim=1).type(torch.float32) # 1, 3, x, y, z

            ego_pcs = []
            for s in tqdm(sensor):
                # load data from NuScenes
                data = nusc.get('sample_data', my_sample['data'][s])

                pose_s2e = nusc.get('calibrated_sensor', data['calibrated_sensor_token'])
                R_pose_s2e = Quaternion(pose_s2e['rotation']).rotation_matrix
                sensor2ego = np.eye(4)
                sensor2ego[:3, :3] = R_pose_s2e
                t_pose_s2e= pose_s2e['translation']
                sensor2ego[:3, 3] = np.array(t_pose_s2e).T
                sensor2ego = torch.from_numpy(sensor2ego).type(torch.float32)

                cam_intrinsic = np.array(pose_s2e['camera_intrinsic'])
                cam_intrinsic = torch.from_numpy(cam_intrinsic.astype(np.float32)).unsqueeze(0) # (1, 3, 3)

                img_pth = os.path.join(nusc.dataroot, data['filename'])
                #print(img_pth)
                #img = iio.imread(img_pth)

                #rgb = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).permute(0, 3, 1, 2) # (1, 3, H, W)
                #rgb = (rgb / 255) * 2 - 1
                
                # get relative depth map
                #rel_depth = get_depth(DA, rgb, use_cuda=True) # (B, 1, H, W)
                depth_pth = os.path.join(nusc_depth_root, data['filename'][:-4]+'_depth.npy')
                depth = torch.from_numpy(np.load(depth_pth)) # (1, 1, H, W)

                # get sematic map
                #sem_map = SAM(img_pth) # np.array (H, W) [-1, 16]
                #sem = torch.from_numpy(sem_map).unsqueeze(0).unsqueeze(0) # (B, 1, H, W)
                
                pts = 0
                sem_pth = os.path.join(nusc_sem_root, data['filename'][:-4]+'_mask.bin')
                if not os.path.exists(sem_pth): 
                    pts = 1
                    break
                sem = np.fromfile(sem_pth, dtype=np.int8).reshape(900, 1600)
                sem = torch.from_numpy(sem).unsqueeze(0).unsqueeze(0)
                
                #mask_rgb = torch.from_numpy(mask_rgb).unsqueeze(0).permute(0,3,1,2) # (B, 3, H, W)

                # img -> cam
                #cam_pc, sampled_rgb = img2pc(rgb, rel_depth, cam_intrinsic, downsample=1, flatten=True) # (1, 4, N) (1, 3, N)
                cam_pc, sampled_sem = img2pc(sem, depth, cam_intrinsic, downsample=1, flatten=True) # (1, 1, N)
                #cam_pc, sampled_rgb = img2pc(mask_rgb, rel_depth, cam_intrinsic, downsample=1, flatten=True) # (1, 4, N) (1, 3, N)

                # cam -> ego
                ego_pc = torch.einsum('x y, b y n->b x n', sensor2ego, cam_pc) # (B, 4, N)
                ego_pc = ego_pc / (ego_pc[:, 3:4, :] + 1e-6) # (B, 4, N) dim_1: x y z 1

                #ego_pcs.append(torch.cat((ego_pc[:, :3, :], (sampled_rgb+1)/2*255), dim=1).squeeze(0).permute(1, 0)) # (N, 6)
                #ego_pcs.append(torch.cat((ego_pc[:, :3, :], sampled_rgb), dim=1).squeeze(0).permute(1, 0)) # (N, 6)
                ego_pcs.append(torch.cat((ego_pc[:, :3, :], sampled_sem), dim=1).squeeze(0).permute(1, 0)) # (N, 4)

                #break

            if not pts == 1: 
                #ego_pcs = torch.cat(ego_pcs, dim=0) # (N, 6)
                ego_pcs = torch.cat(ego_pcs, dim=0) # (N, 4)

                # select points within range
                mask = ((ego_pcs[:,0] > -voxel_range[1]) & (ego_pcs[:,0] < voxel_range[1]) \
                        & (ego_pcs[:,1] > -voxel_range[1]) & (ego_pcs[:,1] < voxel_range[1]) \
                        & (ego_pcs[:,2] < voxel_height[1])).nonzero().flatten()
                ego_pcs = torch.index_select(ego_pcs, 0, mask)
                
                # save pcs
                pcs_folder = os.path.join(pcs_root, my_scene['name'], my_sample['token'])
                if not os.path.exists(pcs_folder): os.makedirs(pcs_folder)
                pcs_filename = os.path.join(pcs_folder, 'pc.npz')
                np.savez(pcs_filename, pc=ego_pcs.detach().numpy())

            # generate occupancy & visualization
            #occ = pc2occ(ego_pcs, (-100, 100), (-10, 40), 1) # (_x, _y, _z)
            #occ_pc = []
            #print(occ.shape)
            #for x in range(occ.shape[0]):
            #    for y in range(occ.shape[1]):
            #        for z in range(occ.shape[2]):
            #            if not occ[x, y, z] == 0:
            #                sem_rgb = torch.from_numpy(colors[int(occ[x, y, z]), :3]).unsqueeze(0) # (1, 3)
            #                xyz = torch.from_numpy(np.array((x, y, z))).unsqueeze(0) # (1, 3)
            #                occ_pc.append(torch.cat((xyz, sem_rgb), dim=1))
            #occ_pcs = torch.cat(occ_pc, dim=0) # (N, 6)

            # save occupancy as .ply file
            #root_dir = 'C:/Users/Wang Jikai/Desktop/occ_data_engine/ply_save/'
            #save_pc_as_ply(occ_pcs.detach().numpy(), root_dir + '{}_occ.ply'.format(sample_count))

            # save pointcloud as .ply file
            #root_dir = 'C:/Users/Wang Jikai/Desktop/data engine for occ/ply_save/'
            #save_pc_as_ply(ego_pcs.detach().numpy(), root_dir + '{}_pc.ply'.format(sample_count))
            
            # next sample
            if my_sample['next'] == '': 
                break
            else:
                my_sample = nusc.get('sample', my_sample['next'])
            #sample_count += 1

            #if sample_count == 2: break


if __name__ == '__main__':
    main((800, 851))