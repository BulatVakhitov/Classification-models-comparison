import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from IPython.display import clear_output


def _plot_metrics(epoch: int, train_acc, val_acc, train_loss, val_loss, title):

    MAX_LOSS = 5
    # Бывает что лосс в начале очень большой, из-за чего всеь остальной график сливается в одну линию.
    # Поэтому если лосс больше чем MAX_LOSS тогда заменяем его на MAX_LOSS

    clear_output(wait=True)

    plt.rcParams["figure.figsize"] = (15, 8)

    plt.subplot(1, 2, 1)
    plt.plot(range(epoch), train_acc, label='train accuracy')
    plt.plot(range(epoch), val_acc, label='validation accuracy')
    plt.title(label=f'{title} accuracy')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    val_loss[-1] = MAX_LOSS if val_loss[-1] > MAX_LOSS else val_loss[-1]
    train_loss[-1] = MAX_LOSS if train_loss[-1] > MAX_LOSS else train_loss[-1]
    plt.plot(range(epoch), train_loss, label='train loss')
    plt.plot(range(epoch), val_loss, label='validation loss')
    plt.title(label=f'{title} loss')
    plt.grid()
    plt.legend()

    plt.show()


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
        for _, (images, labels) in tqdm_train:
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
            tqdm_train.set_postfix(
                loss=np.mean(all_loss),
                train_acc=correct / total
            )

    return (np.mean(all_loss), correct / total)


def _val_epoch(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.CrossEntropyLoss,
               device: torch.device) -> int:

    model.eval()
    total = 0
    correct = 0
    val_loss = []

    with tqdm(
        enumerate(dataloader), total=len(dataloader), desc='Validating'
    ) as tqdm_val:
        with torch.no_grad():
            for _, (images, labels) in tqdm_val:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                predicted = torch.argmax(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss.append(loss.item())
                tqdm_val.set_postfix(
                    val_loss=np.mean(val_loss),
                    val_accuracy=correct / total)

    val_acc = correct / total
    return val_acc, np.mean(val_loss)


def train(model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: Optimizer,
          loss_fn: nn.CrossEntropyLoss,
          device: torch.device,
          scheduler = None,
          n_epochs: int = 100, 
          title: str = '') -> Dict:

    loss_fn = loss_fn.to(device)
    best_state = model.state_dict()
    best_val_acc = 0.0
    train_accuracies = []
    val_accuracies = []
    val_loss = []
    train_loss = []

    for epoch in range(n_epochs):
        print(f'Epoch: [{epoch+1}/{n_epochs}]')

        epoch_train_loss, epoch_train_acc = _train_epoch(
            model, train_dataloader, optimizer, loss_fn, device)

        epoch_val_acc, epoch_val_loss = _val_epoch(
            model, val_dataloader, loss_fn, device)

        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        val_loss.append(epoch_val_loss)
        train_loss.append(epoch_train_loss)

        _plot_metrics(
            epoch + 1,
            train_accuracies,
            val_accuracies,
            train_loss,
            val_loss,
            title
        )

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = model.state_dict()

        print(
            f'Validation accuracy: {epoch_val_acc:.3f}, best_val_acc: {best_val_acc:.3f}')

    test_accuracy = _evaluate(model, best_state, test_dataloader, device)

    print(f'Test accuracy of {title} = {test_accuracy}')

    return {
        'best_state': best_state,
        'best_val_acc': best_val_acc,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies,
        'train_loss': train_loss,
        'val_loss': val_loss,
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
                tqdm_test.set_postfix(accuracy=correct / total)

    accuracy = correct / total
    return accuracy
