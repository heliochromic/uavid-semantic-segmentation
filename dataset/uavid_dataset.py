import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path


class UAVidDataset(Dataset):
    def __init__(self, root_dir, data_part='train', class_mapping=None, transforms=None):
        if data_part not in ['train', 'valid', 'test']:
            raise ValueError('data_part must be train, valid or test')

        if not class_mapping and data_part == 'train':
            raise ValueError('class_mapping is required for train data')

        self.transforms = transforms
        self.data_part = data_part
        self.class_mapping = class_mapping
        self.samples = []
        self.is_test = data_part == 'test'

        if not self.is_test and class_mapping:
            self.rgb_to_class = np.zeros((256, 256, 256), dtype=np.uint8)
            for rgb, class_id in class_mapping.items():
                self.rgb_to_class[rgb[0], rgb[1], rgb[2]] = class_id

        base_path = Path(root_dir) / data_part / data_part

        for seq_dir in sorted(base_path.iterdir()):
            if not seq_dir.is_dir():
                continue

            images_dir = seq_dir / 'Images'

            if not images_dir.exists():
                continue

            for image_path in sorted(images_dir.glob('*.png')):
                sample = {'image_path': str(image_path)}

                if self.is_test:
                    # Test set has no labels
                    self.samples.append(sample)
                else:
                    # Train and valid sets have labels
                    labels_dir = seq_dir / 'Labels'
                    label_path = labels_dir / image_path.name
                    if label_path.exists():
                        sample['mask_path'] = str(label_path)
                        self.samples.append(sample)

        if self.samples:
            self._compute_crop_size()

    def _compute_crop_size(self):
        min_height, min_width = float('inf'), float('inf')

        for sample in self.samples[:10]:
            img = cv2.imread(sample['image_path'])
            h, w = img.shape[:2]
            min_height = min(min_height, h)
            min_width = min(min_width, w)

        self.crop_height = min_height if min_height != float('inf') else None
        self.crop_width = min_width if min_width != float('inf') else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.crop_height and self.crop_width:
            image = image[:self.crop_height, :self.crop_width]

        if self.is_test:
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed['image']
            return image, sample['image_path']

        label = cv2.imread(sample['mask_path'])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if self.crop_height and self.crop_width:
            label = label[:self.crop_height, :self.crop_width]

        mask = self.rgb_to_class[label[:, :, 0],
                                 label[:, :, 1], label[:, :, 2]]

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask