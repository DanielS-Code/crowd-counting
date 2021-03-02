import cv2
import h5py
import torch
import random
import numpy as np
import logging
import argparse

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CrowdEstimationDataset(Dataset):

    def __init__(self, image_paths, transform=None, augment_type=None):
        self.image_paths = image_paths
        self.transform = transform
        self.augment_type = augment_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        # Load density map
        target_path = image_path.replace('.jpg', '.h5').replace('images', 'densities')
        target_file = h5py.File(target_path)
        target = np.asarray(target_file['density'])

        logging.debug(f'input image shape {image.shape}')
        logging.debug(f'input target shape {target.shape}')

        images = [image]  # Default do not perform augmentation
        targets = [target]  # Default do not perform augmentation

        logging.debug(f'augmentation type {self.augment_type}')
        if self.augment_type == 'random_crop':
            images, targets = self.get_random_crop(image, target)
        elif self.augment_type == 'multi_crop':
            images, targets = self.get_multi_crop(image, target)

        logging.debug(f'augmented image shape {images[0].shape}')
        logging.debug(f'augmented target shape {targets[0].shape}')

        # target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64##
        # image = self.transform(image)##

        # Density map must be one eighth of original image
        for i in range(len(targets)):
            width = int(targets[i].shape[1] / 8)
            height = int(targets[i].shape[0] / 8)
            targets[i] = cv2.resize(targets[i], (width, height), interpolation=cv2.INTER_CUBIC) * 64
        if self.transform is not None:
            for i in range(len(images)):
                images[i] = self.transform(images[i])
        image = torch.stack(images)
        target = np.stack(targets)
        logging.debug(f'retruned image shape {image.shape}')
        logging.debug(f'returned target shape {target.shape}')
        return image, target

    def get_random_crop(self, image, target):
        center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
        # Get corner or random patch
        if random.randint(0, 1):
            # corner
            dx = int(random.randint(0, 1) * center[0])
            dy = int(random.randint(0, 1) * center[1])
        else:
            dx = int(random.random() * center[0])
            dy = int(random.random() * center[1])
        cropped_image = image[dy: dy + center[1], dx: dx + center[0]]
        cropped_target = target[dy: dy + center[1], dx: dx + center[0]]
        if random.randint(0, 1):
            cropped_image = np.fliplr(cropped_image) - np.zeros_like(cropped_image)
            cropped_target = np.fliplr(cropped_target) - np.zeros_like(cropped_target)
        return [cropped_image], [cropped_target]

    def get_crop(self, image, target, corner1, corner2, flip = False):
        center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
        # Choose corners
        dx = int(corner1 * center[0])
        dy = int(corner2 * center[1])
        cropped_image = image[dy: dy + center[1], dx: dx + center[0]]
        cropped_target = target[dy: dy + center[1], dx: dx + center[0]]
        if flip:
            cropped_image = np.fliplr(cropped_image) - np.zeros_like(cropped_image)
            cropped_target = np.fliplr(cropped_target) - np.zeros_like(cropped_target)
        return cropped_image, cropped_target

    def get_multi_crop(self, image, target):
        images = []
        targets = []
        for i in range(2):
            for j in range(2):
                cropped_image, cropped_target = self.get_crop(image,target,i,j)
                images.append(cropped_image)
                targets.append(cropped_target)
                cropped_image, cropped_target = self.get_crop(image, target, i, j, flip = True)
                images.append(cropped_image)
                targets.append(cropped_target)
        return images, targets

def create_dataloader(image_paths, shuffle = True, augment_type=None):
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformed_dataset = CrowdEstimationDataset(image_paths, augment_type=augment_type,
                                                 transform=preprocessing)
    return DataLoader(transformed_dataset, shuffle=shuffle)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to train.json')
    parser.add_argument('target_path', type=str, help='Path to val.json')
    parser.add_argument('--augment_type', '-a', type=str, default=None,
                        help='Augmentation type')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])