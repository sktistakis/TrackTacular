import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from datasets.medtrack_dataset import MedTrack
from datasets.pedMedTrack_dataset import PedMedTrackDataset
from datasets.sampler import TemporalSampler


class PedMedTrackDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MedTrack",
            recordings_train=None,
            recordings_val=None,
            recordings_test=None,
            batch_size: int = 2,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            accumulate_grad_batches=8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.recordings_train = recordings_train or []
        self.recordings_val = recordings_val or []
        self.recordings_test = recordings_test or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        self.dataset = os.path.basename(self.data_dir)

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    def setup(self, stage: Optional[str] = None):
        
        if stage == 'fit':
            self.data_train = PedMedTrackDataset(
                MedTrack(self.data_dir, self.recordings_train),
                is_train=True,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = PedMedTrackDataset(
                MedTrack(self.data_dir, self.recordings_val),
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'test':
            self.data_test = PedMedTrackDataset(
                MedTrack(self.data_dir, self.recordings_test),
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'predict':
            self.data_predict = PedMedTrackDataset(
                MedTrack,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_train, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_val, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )