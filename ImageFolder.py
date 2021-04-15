# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
import os

class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, ratio=1):
        # splitdir = Path(root)
        self.samples = []
        if ratio > 1:
            raise RuntimeError(f'Invalid Data Ratio "{ratio}"')
        for r, dirs, files in os.walk(root):
            mini_samples = []
            for f in files:
                if f.split(".")[1] in ["jpg", "jpeg", "png"]:
                    mini_samples.append(Path(os.path.join(r, f)))
            if len(mini_samples) <= 0:
                continue
            mini_samples = random.sample(mini_samples, int(len(mini_samples) * ratio))
            self.samples.extend(mini_samples)

        # if not splitdir.is_dir():
        #     raise RuntimeError(f'Invalid directory "{root}"')


        # self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.samples = np.random.choice(self.samples, int(len(self.samples)), replace=False)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        try:
            img = Image.open(self.samples[index]).convert("RGB")
            if self.transform:
                return self.transform(img)
        except Exception:
            return self.transform(Image.fromarray(np.uint8(np.zeros([4096, 4096, 3]))))
        return img

    def __len__(self):
        return len(self.samples)


class TinyImageNet(Dataset):
    def __init__(self, img, transform=None):
        self.img = img.transpose((0, 2, 3, 1))
        self.transform = transform
        self.len = self.img.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        img = (Image.fromarray((self.img[index]).astype(np.uint8)))

        if self.transform is not None:
            img = self.transform(img)
        return img