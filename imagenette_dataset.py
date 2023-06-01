import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict, List
from torch.utils.data import random_split
import os
import random
import matplotlib.pyplot as plt


def train_val_split(dataset: Dataset, val_size: int) -> Tuple[Subset, Subset]:
    total_samples = len(dataset)

    n_train_samples = int((1 - val_size) * total_samples)
    n_val_samples = total_samples - n_train_samples
    train_dataset, val_dataset = random_split(dataset, [n_train_samples, n_val_samples])

    return (train_dataset, val_dataset)


class ImageNetteDataset(Dataset):

    def __init__(self, root_dir: str=None, subset: Subset=None) -> None:
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.data = []

        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            for image in os.listdir(class_path):
                if image.startswith('.'):
                    os.remove(os.path.join(class_path, image))
                else:
                    img_path = os.path.join(class_path, image)
                    self.data.append((img_path, class_name))

        self._class_name_to_normal_name = {
            'n03000684': 'chainsaw',
            'n03417042': 'garbacge_track',
            'n03425413': 'gas pump',
            'n02102040': 'English Springer(dog)',
            'n03028079': 'church',
            'n01440764': 'tench(fish)',
            'n02979186': 'cassete player',
            'n03888257': 'parachute',
            'n03394916': 'french horn',
            'n03445777': 'golf ball'
        }

        self._class_name_to_index = {
            'n03000684': 0,
            'n03417042': 1,
            'n03425413': 2,
            'n02102040': 3,
            'n03028079': 4,
            'n01440764': 5,
            'n02979186': 6,
            'n03888257': 7,
            'n03394916': 8,
            'n03445777': 9
        }

    def load_image(self, index: int) -> Tuple[Image.Image, str]:
        image_path, class_name = self.data[index]
        return Image.open(image_path), class_name

    def random_data_examples(self) -> None:
        class_img_examples = []

        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)

            image = random.choice(os.listdir(class_path))
            img_path = os.path.join(class_path, image)

            class_img_examples.append((Image.open(img_path), class_name))

        fig = plt.figure(figsize=(14, 7))
        columns = 5
        rows = 2

        for i in range(columns*rows):
            image, class_name = class_img_examples[i]
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(image)
            plt.title(self._class_name_to_normal_name[class_name])
        plt.show()

    def __len__(self) -> int:
        return(len(self.data))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, class_name = self.load_image(index)
        img = img.convert("RGB")
        img = self.transform(img)

        return img, self._class_name_to_index[class_name]