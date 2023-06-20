import os
import numpy as np
from typing import Tuple, Dict, List
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


def get_random_images():
    class_img_examples = []

    for class_name in os.listdir('data/imagenette2/val'):
        class_path = os.path.join('data/imagenette2/val', class_name)

        image = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, image)

        class_img_examples.append((Image.open(img_path), class_name))

    return class_img_examples

def plot_predictions(model: nn.Module, 
                     class_img_examples: List, 
                     device: torch.device):

    class_name_to_normal_name = {
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

    index2normal_name = {
             0:'chainsaw',
             1:'garbage_track',
             2:'gas pump',
             3:'English Springer(dog)',
             4:'church',
             5:'tench(fish)',
             6:'cassete player',
             7:'parachute',
             8:'french horn',
             9:'golf ball'
        }

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    fig = plt.figure(figsize=(17, 10))
    columns = 5
    rows = 2

    model.eval()
    with torch.no_grad():
        for i, (image, class_name) in enumerate(class_img_examples):
            image = image.convert("RGB")
            image = transform(image)
            image = image.to(device)
            output = model(image[None, :])
            predicted = torch.argmax(output.data, dim=1).item()
            prob = nn.functional.softmax(output, dim=1)
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(class_img_examples[i][0])
            plt.title(label=f'ground truth: {class_name_to_normal_name[class_name]}, \npredicted: {index2normal_name[predicted]}, \np = {prob[0][predicted]}')
        plt.show()


def plot_confusion_matrix(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()

    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs.data, dim=1)

            all_predicted.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        cm = confusion_matrix(all_predicted, all_labels)
        ConfusionMatrixDisplay(cm).plot()