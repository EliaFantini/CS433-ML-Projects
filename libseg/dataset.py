import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from libseg.preprocessing import preprocess, data_augmentation_test, preprocess_test


class ToTensor(object):
    def __init__(self, device: str = None):
        """
        Parameters
        ----------
        device: str
            move to device if specified
        """
        self.device = device

    def img_to_tensor(self, img: np.ndarray, config):
        """
        Transforms an array of pixel representing an image into a tensor.
        Values are scaled from 0-255 to 0-1.
        Parameters
        ----------
        img: np.ndarray
            ndarray of RGB values
        config: dict
        Dictionary containing all execution's parameters
        return: tensor
        """
        img = img.transpose((2, 0, 1))
        tensor = torch.Tensor(img)
        if not config['normalize_and_centre']:
            tensor /= 255
        tensor.unsqueeze(0)
        if self.device:
            tensor = tensor.to(self.device)  # it must be assigned otherwise it doesn't change
        return tensor

    def mask_to_tensor(self, mask: np.ndarray):
        """
        Transforms an array of ground truth values into a tensor.
        Parameters
        ----------
        mask: np.ndarray
            ndarray of  ground truth values. It should contain just 0-1 values, where 0 is background.
        return: tensor
        """
        tensor = transforms.ToTensor()(mask)
        tensor = torch.round(tensor)
        # tensor.unsqueeze(0)
        if self.device:
            tensor = tensor.to(self.device)  # it must be assigned otherwise it doesn't change
        return tensor[0, :, :][None, :, :]


class SegmentDataset(Dataset):

    def __init__(self,
                 img_paths,
                 gt_paths,
                 config,
                 target = True,
                 mode = 'train'):
        """

        @param img_paths: list
            Paths to images' files
        @param gt_paths: list
            Paths to ground truth' files
        @param config: dict
            Dictionary containing all execution's parameters
        @param target: bool
            Set to True if gt_paths were provided
        @param mode: str
            Set to "train" if the dataset will be used to train the model,
            "test" if it is used to predict test data.
        """
        self.config = config
        self.mode = mode
        self.target = target
        to_tensor = ToTensor()

        if mode == 'train':
            images, groundtruths = self._load_data(img_paths, gt_paths)
            images, groundtruths = preprocess(images, groundtruths, self.config)
            self.images = [to_tensor.img_to_tensor(img, config) for img in images]
            self.gt = [to_tensor.mask_to_tensor(gt) for gt in groundtruths]
        else:
            if not self.target:
                images = self._load_data(img_paths)
            else:
                images, groundtruths = self._load_data(img_paths, gt_paths)
                self.gt = [to_tensor.mask_to_tensor(gt) for gt in groundtruths]

            images = preprocess_test(images, config)
            images = data_augmentation_test(images)
            images = np.array([np.array(image) for image in images])
            self.images = [to_tensor.img_to_tensor(img, config) for img in images]


    def __getitem__(self, i):
        if self.mode == 'train':
            return self.images[i], self.gt[i]
        else:
            return self.images[i]

    def __len__(self):
        return len(self.images)

    def _load_data(self, paths_img,
                   paths_gt=None):
        """
        Inner function to load data as PIL images using provided paths
        @param img_paths: list
            Paths to images' files
        @param gt_paths: list
            Paths to ground truth' files
        @return: list
            Return list of PIL images. If paths_gt is not None, a list of ground truth images as PIL Image
            is returned as well.
        """
        images = [Image.open(image) for image in paths_img]

        if self.target:
            gt = [Image.open(target) for target in paths_gt]
            return images, gt
        else:
            return images
