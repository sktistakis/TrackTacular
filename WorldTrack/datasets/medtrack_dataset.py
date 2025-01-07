import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset

intrinsic_camera_matrix_filenames = ['int_Cam0.xml', 'int_Cam1.xml', 'int_Cam2.xml', 'int_Cam3.xml',
                                     'int_Cam4.xml', 'int_Cam5.xml']
extrinsic_camera_matrix_filenames = ['extr_Cam0.xml', 'extr_Cam1.xml', 'extr_Cam2.xml', 'extr_Cam3.xml',
                                     'extr_Cam4.xml', 'extr_Cam5.xml']


class MedTrack(VisionDataset):
    def __init__(self, root, recordings):
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indeing; x,y (w,h) (n_col,n_row)
        # MedTrack has ij-indexing: H*W=480*320, thus x (i) is \in [0,480), y (j) is \in [0,320)
        # MedTrack has in-consistent unit: centi-meter (cm) for calibration & pos annotation
        self.__name__ = 'MedTrack'
        self.recordings = recordings
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 320]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 6, 700
        self.frame_step = 1
        # world x,y actually means i,j in Wildtrack, which correspond to h,w
        self.worldcoord_from_worldgrid_mat = np.array([[0, 2.5, -300], [2.5, 0, -900], [0, 0, 1]])

        self.num_frames_per_recording = {recording: self.get_num_frames(recording) for recording in recordings}
        self.img_fpaths = self.get_image_fpaths()
        
        # Initialize camera matrices for all recordings
        self.intrinsic_matrices = []
        self.extrinsic_matrices = []
        for recording in recordings:
            intrinsic_per_recording = []
            extrinsic_per_recording = []
            for cam in range(self.num_cam):
                intrinsic, extrinsic = self.get_intrinsic_extrinsic_matrix(cam, recording)
                intrinsic_per_recording.append(intrinsic)
                extrinsic_per_recording.append(extrinsic)
            self.intrinsic_matrices.append(np.array(intrinsic_per_recording))
            self.extrinsic_matrices.append(np.array(extrinsic_per_recording))

    def get_image_fpaths(self):
        """
        Generate a dictionary of file paths for images from multiple recordings.
        Returns: {recording_id: {cam: {frame: filepath}}}
        """
        img_fpaths = {}
        
        for recording_id, recording in enumerate(self.recordings):
            recording_path = os.path.join(self.root, recording)
            img_fpaths[recording_id] = {cam: {} for cam in range(self.num_cam)}
            
            for camera_folder in sorted(os.listdir(os.path.join(recording_path, 'Image_subsets'))):
                cam = int(camera_folder[-1])
                if cam >= self.num_cam:
                    continue
                
                camera_path = os.path.join(recording_path, 'Image_subsets', camera_folder)
                for fname in sorted(os.listdir(camera_path)):
                    frame = int(fname.split('.')[0])
                    img_fpaths[recording_id][cam][frame] = os.path.join(camera_path, fname)
        
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_y = pos % 480
        grid_x = pos // 480
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i, recording):
        recording_path = os.path.join(self.root, recording, 'calibrations')

        # Load intrinsic matrix
        intrinsic_file = os.path.join(recording_path, 'intrinsic_zero',
                                    intrinsic_camera_matrix_filenames[camera_i])
        intrinsic_params_file = cv2.FileStorage(intrinsic_file, flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        # Load extrinsic matrix
        extrinsic_file = os.path.join(recording_path, 'extrinsic',
                                    extrinsic_camera_matrix_filenames[camera_i])
        extrinsic_params_file_root = ET.parse(extrinsic_file).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.strip().split(' ')
        rvec = np.array(list(map(float, rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.strip().split(' ')
        tvec = np.array(list(map(float, tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    def get_num_frames(self, recording):
        cam0_path = os.path.join(self.root, recording, 'Image_subsets', 'C0')
        return len(os.listdir(cam0_path))