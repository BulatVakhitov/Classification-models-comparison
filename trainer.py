import numpy as np
from typing import Tuple, Dict, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from IPython.display import clear_output


def _train_epoch(model: nn.Module, dataloader: DataLoader,
                 optimizer: Optimizer,
                 loss_fn: nn.CrossEntropyLoss,
                 device: torch.device) -> Tuple[int, int]:
    all_loss = []
    total = 0
    correct = 0
    model.train()
    with tqdm(
        enumerate(dataloader), total=len(dataloader), desc='Training'
    ) as tqdm_train:
        for i, (images, labels) in tqdm_train:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            tqdm_train.set_postfix(loss=np.mean(all_loss), train_acc=correct/total)
    return (np.mean(all_loss), correct/total)


def _val_epoch(model: nn.Module, 
               dataloader: DataLoader, 
               device: torch.device) -> int:

    model.eval()
    total = 0
    correct = 0

    with tqdm(
        enumerate(dataloader), total=len(dataloader), desc='Validating'
    ) as tqdm_val:
        with torch.no_grad():
            for i, (images, labels) in tqdm_val:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predicted = torch.argmax(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tqdm_val.set_postfix(val_accuracy=correct/total)

    val_acc = correct / total
    return val_acc


def train(model: nn.Module, 
          train_dataloader: DataLoader, 
          val_dataloader: DataLoader, 
          test_dataloader: DataLoader,
          optimizer: Optimizer, 
          loss_fn: nn.CrossEntropyLoss,
          device: torch.device,
          scheduler=None, 
          n_epochs: int=100, 
          title: str='') -> Dict:

    best_state = model.state_dict()
    best_val_acc = 0.0
    train_accuracies = []
    val_accuracies = []
    all_loss = []

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}]')

        epoch_loss, epoch_train_acc = _train_epoch(model, train_dataloader, optimizer, loss_fn, device)

        epoch_val_acc = _val_epoch(model, val_dataloader, device)

        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        all_loss.append(epoch_loss)

        clear_output(wait=True)
        plt.plot(range(epoch + 1), val_accuracies, label='validation accuracy')
        plt.plot(range(epoch + 1), train_accuracies, label='train accuracy')
        plt.title(label=title)
        plt.grid()
        plt.legend()
        plt.show()

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): 
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = model.state_dict()

        print(f'Validation accuracy: {epoch_val_acc:.3f}, best_val_acc: {best_val_acc:.3f}')

    test_accuracy = _evaluate(model, best_state, test_dataloader, device)

    print(f'Test accuracy of {title} = {test_accuracy}')

    return {
        'best_state': best_state,
        'best_val_acc': best_val_acc,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies,
        'all_loss': all_loss,
        'test_acc': test_accuracy
    }


def _evaluate(model: nn.Module, best_state: Dict, dataloader: DataLoader, device: torch.device) -> int:
    model.load_state_dict(best_state)
    model.eval()
    total = 0
    correct = 0

    with tqdm(
        enumerate(dataloader), total=len(dataloader), desc='Testing'
    ) as tqdm_test:
        with torch.no_grad():
            for i, (images, labels) in tqdm_test:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predicted = torch.argmax(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tqdm_test.set_postfix(accuracy=correct/total)

    accuracy = correct / total
    return accuracy