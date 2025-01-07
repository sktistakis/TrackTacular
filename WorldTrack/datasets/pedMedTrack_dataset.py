import os
import json
from operator import itemgetter

import torch
import numpy as np
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image

from utils import geom, basic, vox


class PedMedTrackDataset(VisionDataset):
    def __init__(
            self,
            base,
            is_train=True,
            resolution=(160, 4, 250),
            bounds=(-500, 500, -320, 320, 0, 2),
            final_dim: tuple = (720, 1280),
            resize_lim: list = (0.8, 1.2),
    ):
        super().__init__(base.root)
        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        # img_shape and worldgrid_shape is the original shape matching the annotations in dataset
        # MultiviewX: [1080, 1920], [640, 1000] Wildtrack: [1080, 1920], [480, 1440]
        self.img_shape = base.img_shape
        self.worldgrid_shape = base.worldgrid_shape
        self.is_train = is_train
        self.bounds = bounds
        self.resolution = resolution
        self.data_aug_conf = {'final_dim': final_dim, 'resize_lim': resize_lim}
        self.kernel_size = 1.5
        self.max_objects = 60
        self.img_downsample = 4

        self.Y, self.Z, self.X = self.resolution
        self.scene_centroid = torch.tensor((0., 0., 0.)).reshape([1, 3])

        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False)
        
        self.img_fpaths = self.base.get_image_fpaths()
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.download()

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        self.prepare_gt()

        self.calibration = {}
        self.setup()

        # Generate metadata for global index to recording and frame mapping
        self.metadata = []
        for recording_id, recording in enumerate(self.base.recordings):
            num_frames = self.base.num_frames_per_recording[recording]  # Predefined number of frames per recording
            self.metadata.extend([(recording_id, frame) for frame in range(num_frames)])


    def setup(self):
        # Stack all intrinsic and extrinsic matrices from all recordings
        intrinsic_matrices = [torch.tensor(np.stack(recording_intrinsics, axis=0), dtype=torch.float32)
                            for recording_intrinsics in self.base.intrinsic_matrices]
        extrinsic_matrices = [torch.tensor(np.stack(recording_extrinsics, axis=0), dtype=torch.float32)
                            for recording_extrinsics in self.base.extrinsic_matrices]

        # Merge intrinsics for each recording
        intrinsic = [geom.merge_intrinsics(*geom.split_intrinsics(recording)).squeeze()
                    for recording in intrinsic_matrices]

        # Prepare calibration data for all recordings
        self.calibration['intrinsic'] = intrinsic
        self.calibration['extrinsic'] = [torch.eye(4)[None].repeat(recording.shape[0], 1, 1) for recording in intrinsic]
        for i, extrinsics in enumerate(extrinsic_matrices):
            self.calibration['extrinsic'][i][:, :3] = extrinsics

    def prepare_gt(self):
        og_gt = []
        for recording in self.base.recordings:
            recording_path = os.path.join(self.root, recording)
            for fname in sorted(os.listdir(os.path.join(recording_path, 'annotations_positions'))):
                frame = int(fname.split('.')[0])
                with open(os.path.join(recording_path, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                for single_pedestrian in all_pedestrians["root"].values():
                    def is_in_cam(cam):
                        return not (single_pedestrian['views'][str(cam)]['xmin'] == -1 and
                                    single_pedestrian['views'][str(cam)]['xmax'] == -1 and
                                    single_pedestrian['views'][str(cam)]['ymin'] == -1 and
                                    single_pedestrian['views'][str(cam)]['ymax'] == -1)

                    in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                    if not in_cam_range:
                        continue
                    if single_pedestrian['positionsID'] is None:
                        continue
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionsID'])
                    og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self):
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        for recording_id, recording  in enumerate(self.base.recordings):
            self.imgs_gt[recording_id] = {}
            self.world_gt[recording_id] = {}
            recording_path = os.path.join(self.root, recording)
            for fname in sorted(os.listdir(os.path.join(recording_path, 'annotations_positions'))):
                frame = int(fname.split('.')[0])
                num_frame += 1
                with open(os.path.join(recording_path, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]

                for pedestrian in all_pedestrians["root"].values():
                    if pedestrian['positionsID'] == None:  # TODO: shouldnt have Nones. Check dataset creation code.
                        continue
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionsID']).squeeze()
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(pedestrian['personID'])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][str(cam)]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                (pedestrian['views'][str(cam)]))
                            img_pids[cam].append(pedestrian['personID'])
                            num_imgs_bbox += 1
                self.world_gt[recording_id][frame] = (torch.tensor(world_pts, dtype=torch.float32),
                                        torch.tensor(world_pids, dtype=torch.float32))
                self.imgs_gt[recording_id][frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[recording_id][frame][cam] = (torch.tensor(img_bboxs[cam]), torch.tensor(img_pids[cam]))

    def get_bev_gt(self, mem_pts, mem_pts_prev, pids, pids_pre):
        center = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)
        offset = torch.zeros((4, self.Y, self.X), dtype=torch.float32)
        person_ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)

        prev_pts = dict(zip(pids_pre.int().tolist(), mem_pts_prev[0]))

        for pts, pid in zip(mem_pts[0], pids):
            ct = pts[:2]
            ct_int = ct.int()

            if ct_int[0] < 0 or ct_int[0] >= self.X or ct_int[1] < 0 or ct_int[1] >= self.Y:
                continue

            for c in center:
                basic.draw_umich_gaussian(c, ct_int, self.kernel_size)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:2, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid

            if pid in pids_pre:
                t_off = prev_pts[pid.int().item()][:2] - ct_int
                if t_off.abs().max() > 15:
                    continue
                offset[2:, ct_int[1], ct_int[0]] = t_off

        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, sx, sy, crop):
        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)
        center = torch.zeros((3, H, W), dtype=torch.float32)
        offset = torch.zeros((2, H, W), dtype=torch.float32)
        size = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        center_pts = np.stack(((xmin + xmax) / 2, (ymin + ymax) / 2), axis=1)
        center_pts = torch.tensor(center_pts, dtype=torch.float32)
        size_pts = np.stack(((-xmin + xmax), (-ymin + ymax)), axis=1)
        size_pts = torch.tensor(size_pts, dtype=torch.float32)
        foot_pts = np.stack(((xmin + xmax) / 2, ymin), axis=1)
        foot_pts = torch.tensor(foot_pts, dtype=torch.float32)
        head_pts = np.stack(((xmin + xmax) / 2, ymax), axis=1)
        head_pts = torch.tensor(head_pts, dtype=torch.float32)

        for pt_idx, (pid, wh) in enumerate(zip(img_pids, size_pts)):
            for idx, pt in enumerate((foot_pts[pt_idx], )):  # , center_pts[pt_idx], head_pts[pt_idx])):
                if pt[0] < 0 or pt[0] >= W or pt[1] < 0 or pt[1] >= H:
                    continue
                basic.draw_umich_gaussian(center[idx], pt.int(), self.kernel_size)

            ct_int = foot_pts[pt_idx].int()
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = foot_pts[pt_idx] - ct_int
            size[:, ct_int[1], ct_int[0]] = wh
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, offset, size, person_ids, valid_mask

    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_image_data(self, frame, cameras, recording_id):
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []
        
        for cam in cameras:
            # Use the recording ID to get the correct file paths
            img_path = self.base.img_fpaths[recording_id][cam][frame]
            img = Image.open(img_path).convert('RGB')
            
            W, H = img.size
            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            # Fetch intrinsic and extrinsic calibrations for the specific camera and recording
            extrin = self.calibration['extrinsic'][recording_id][cam]
            intrin = self.calibration['intrinsic'][recording_id][cam]
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))
            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]
            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = basic.img_transform(img, resize_dims, crop)
            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            # Process ground truth annotations for this frame
            if cam in self.imgs_gt[recording_id][frame] and self.imgs_gt[recording_id][frame][cam][0].size(0) > 0:
                img_pts, img_pids = self.imgs_gt[recording_id][frame][cam]
                center_img, offset_img, size_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids, sx, sy, crop)
            else:
                # Use empty tensors if no data is available for this camera
                H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
                W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)
                center_img = torch.zeros((3, H, W), dtype=torch.float32)
                offset_img = torch.zeros((2, H, W), dtype=torch.float32)
                size_img = torch.zeros((2, H, W), dtype=torch.float32)
                valid_img = torch.zeros((1, H, W), dtype=torch.bool)
                pid_img = torch.zeros((1, H, W), dtype=torch.long)

            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            pids.append(pid_img)
            valids.append(valid_img)

        return (
            torch.stack(imgs), torch.stack(intrins), torch.stack(extrins),
            torch.stack(centers), torch.stack(offsets), torch.stack(sizes),
            torch.stack(pids), torch.stack(valids)
        )

    def __len__(self):
        return sum(self.base.num_frames_per_recording.values()) - 1

    def __getitem__(self, index):
        # Use metadata to map the global index to recording and frame
        recording_id, frame = self.metadata[index]
        recording = self.base.recordings[recording_id]

        # Determine the previous frame (handle edge case for the first frame)
        pre_frame = max(frame - 1, 0)

        # Retrieve data paths and calibrations for the specific recording
        cameras = list(range(self.num_cam))
        imgs, intrins, extrins, centers_img, offsets_img, sizes_img, pids_img, valids_img = self.get_image_data(
            frame, cameras, recording_id
        )

        # Calibration transformation matrices
        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(self.base.worldcoord_from_worldgrid_mat, dtype=torch.float32)
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)

        # Fetch ground truth for the current and previous frames
        worldgrid_pts_org, world_pids = self.world_gt[recording_id][frame]
        worldgrid_pts_pre, world_pid_pre = self.world_gt[recording_id][pre_frame]

        # Prepare world grid points
        worldgrid_pts = torch.cat((worldgrid_pts_org, torch.zeros_like(worldgrid_pts_org[:, 0:1])), dim=1).unsqueeze(0)
        worldgrid_pts_pre = torch.cat((worldgrid_pts_pre, torch.zeros_like(worldgrid_pts_pre[:, 0:1])), dim=1)

        # Apply augmentations during training
        if self.is_train:
            Rz = torch.eye(3)
            scene_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            off = 0.25
            scene_center[:2].uniform_(-off, off)
            augment = geom.merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
            worldgrid_T_worldcoord = torch.matmul(augment, worldgrid_T_worldcoord)
            worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), worldgrid_pts)

        # Convert world grid points to memory points
        mem_pts = self.vox_util.Ref2Mem(worldgrid_pts, self.Y, self.Z, self.X)
        mem_pts_pre = self.vox_util.Ref2Mem(worldgrid_pts_pre.unsqueeze(0), self.Y, self.Z, self.X)
        center_bev, valid_bev, pid_bev, offset_bev = self.get_bev_gt(mem_pts, mem_pts_pre, world_pids, world_pid_pre)

        # Prepare grid ground truth
        grid_gt = torch.zeros((self.max_objects, 3), dtype=torch.long)
        grid_gt[:worldgrid_pts.shape[1], :2] = worldgrid_pts_org
        grid_gt[:worldgrid_pts.shape[1], 2] = world_pids

        # Create item and target dictionaries
        item = {
            'img': imgs,  # S,3,H,W
            'intrinsic': intrins,  # S,4,4
            'extrinsic': extrins,  # S,4,4
            'ref_T_global': worldgrid_T_worldcoord,  # 4,4
            'frame': frame,
            'sequence_num': recording_id,
            'grid_gt': grid_gt,
        }

        target = {
            # BEV (bird's-eye view)
            'valid_bev': valid_bev,  # 1,Y,X
            'center_bev': center_bev,  # 1,Y,X
            'offset_bev': offset_bev,  # 2,Y,X
            'pid_bev': pid_bev,  # 1,Y,X
            # Image
            'center_img': centers_img,  # S,1,H/8,W/8
            'offset_img': offsets_img,  # S,2,H/8,W/8
            'size_img': sizes_img,  # S,2,H/8,W/8
            'valid_img': valids_img,  # S,1,H/8,W/8
            'pid_img': pids_img,  # S,1,H/8,W/8
        }

        return item, target

