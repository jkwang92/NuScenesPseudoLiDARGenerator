import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from plyfile import PlyData, PlyElement
import cv2

import open3d as o3d
from open3d.pipelines.registration import compute_fpfh_feature


def img2pc(img, depth, cam_intrinsic, downsample=1, flatten=True):
    # img: torch.Tensor (B, 3, H, W) [-1, 1]
    # depth: torch.Tensor (B, 1, H, W) # suppose input "depth" is relative depth
    # cam_intrinsic: torch.Tensor (B, 3, 3)
    device = img.device

    fx, fy, cx, cy = cam_intrinsic[:, 0, 0], cam_intrinsic[:, 1, 1], cam_intrinsic[:, 0, 2], cam_intrinsic[:, 1, 2] # (B, )

    # generate image coordinate
    hds, wds = downsample if isinstance(downsample, tuple) else downsample, downsample
    b, _, h, w = depth.shape
    mesh = torch.meshgrid(torch.arange(0, h, hds), torch.arange(0, w, wds))
    coord = torch.cat([m.unsqueeze(0) for m in mesh], dim=0).type(torch.float32).unsqueeze(0).expand(b, -1, -1, -1).to(device) # (B, 2, h, w) h=H//downsample w=W//downsample

    sampled_depth = depth[:, :, 0:h:hds, 0:w:wds] # (B, 1, h, w)
    sampled_rgb = img[:, :, 0:h:hds, 0:w:wds] # (B, 3, h, w)

    # image system to camera system
    #cam_z = fx / (sampled_depth + 1e-6) # (B, 1, h, w) # relative depth to depth
    cam_z = sampled_depth
    cam_x = (coord[:, 1:2, :, :] - cx) * cam_z / fx # (B, 1, h, w)
    cam_y = (coord[:, 0:1, :, :] - cy) * cam_z / fy # (B, 1, h, w)
    cam_n = torch.ones_like(cam_z)
    cam_pc = torch.cat((cam_x, cam_y, cam_z, cam_n), dim=1).type(torch.float32) # (B, 4, h, w) # dim_1: x y z 1

    if flatten: 
        cam_pc = cam_pc.flatten(2) # (B, 4, N)
        sampled_rgb = sampled_rgb.flatten(2) # (B, 3, N)

    return cam_pc, sampled_rgb


def pc2occ(pc, xy_range=(-100, 100), z_range=(-10, 40), res=1):

    voxel_coord_x = torch.arange(0, (xy_range[1]-xy_range[0])//res)
    voxel_coord_y = torch.arange(0, (xy_range[1]-xy_range[0])//res)
    voxel_coord_z = torch.arange(0, (z_range[1]-z_range[0])//res)
    voxel_coord = torch.meshgrid(voxel_coord_x, voxel_coord_y, voxel_coord_z)
    coord = torch.cat([d.unsqueeze(0).unsqueeze(0) for d in voxel_coord], dim=1).type(torch.float32) # 1, 3, x, y, z

    pc[:, :2] = ((pc[:, :2] - xy_range[0]) / (xy_range[1] - xy_range[0])) * 2 - 1
    pc[:, 2] = ((pc[:, 2] - z_range[0]) / (z_range[1] - z_range[0])) * 2 - 1

    pc_coord = pc[:, :3]
    pc_label = pc[:, 3]

    occ_coord = F.grid_sample(coord, pc_coord.view(1, 1, 1, -1, 3).flip(-1).type(torch.float32), mode='nearest', align_corners=True)[:, :, 0, 0, :]\
        .permute(0, 2, 1).squeeze(0).detach().numpy() # N 3

    occ = torch.zeros(coord.shape[-3:])

    for i in range(occ_coord.shape[0]):
        idx, idy, idz = int(occ_coord[i, 0]), int(occ_coord[i, 1]), int(occ_coord[i, 2])
        occ[idx, idy, idz] = pc_label[i]

    return occ


def get_orb_descriptor(img, mask=None):
    # img: np.array (H, W, 3) [0,255]

    # rgb to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

    # generate ORB descriptor
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, mask)

    kp_x = np.expand_dims(np.array([p.pt[0] for p in kp]), axis=-1) # (N, 1)
    kp_y = np.expand_dims(np.array([p.pt[1] for p in kp]), axis=-1) # (N, 1)
    kp = np.concatenate((kp_x, kp_y), axis=-1) # (N, 2)
    des = np.array(des) # (N, C)

    return kp, des # (N, 2), (N, C)


def pc_alignment(pc, descriptor=None):
    # pc: tuple(np.array, np.array) ((N, 3), (N, 3))
    # descriptor: tuple(np.array, np.array ((N, C), (N, C))

    # prepare pointcloud as "o3d.geometry.PointCloud" class
    pc1, pc2 = pc
    src_pc = o3d.geometry.PointCloud()
    tgt_pc = o3d.geometry.PointCloud()
    src_pc.points = o3d.utility.Vector3dVector(pc1)
    tgt_pc.points = o3d.utility.Vector3dVector(pc2)

    # prepare descriptor as "o3d.registration.Feature" class
    # use customized descriptor
    if not descriptor == None:
        des1, des2 = descriptor
        src_feat = o3d.pipelines.registration.Feature()
        tgt_feat = o3d.pipelines.registration.Feature()
        src_feat.data = des1.T
        tgt_feat.data = des2.T

    # FPFH descriptor
    else:
        src_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        tgt_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        src_feat = compute_fpfh_feature(src_pc, search_param)
        tgt_feat = compute_fpfh_feature(tgt_pc, search_param)

    # set RANSAC parameters
    max_correspondence_distance = 0.05
    ransac_n = 3
    checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)

    # RANSAC alignment
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=src_pc, target=tgt_pc,
        source_feature=src_feat, target_feature=tgt_feat,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n, checkers=checkers, criteria=criteria)

    src2tgt = ransac_result.transformation # np.ndarray (4, 4)

    return src2tgt


def save_pc_as_ply(pc, pth):
    # points: np.array (N, 3)/(N, 6)
    
    points = np.array(pc)

    if pc.shape[1] == 6:
        vertex = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]
        vertex['red'] = points[:, 3]
        vertex['green'] = points[:, 4]
        vertex['blue'] = points[:, 5]

    elif pc.shape[1] == 3:
        vertex = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]

    ply_data = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    ply_data.write(pth)


if __name__ == '__main__':

    import imageio as iio
    import matplotlib.pyplot as plt

    cam_front       = iio.imread('C:/Users/Wang Jikai/Desktop/data engine for Occ/v1.0-mini\samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg')
    cam_front_right = iio.imread('C:/Users/Wang Jikai/Desktop/data engine for Occ/v1.0-mini\samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg')
    
    kp_f, des_f = get_orb_descriptor(cam_front)
    kp_fr, des_fr = get_orb_descriptor(cam_front_right)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(cam_front)
    ax1.scatter(kp_f[:,0], kp_f[:,1])

    ax2.imshow(cam_front_right)
    ax2.scatter(kp_fr[:,0], kp_fr[:,1])
    plt.show()

