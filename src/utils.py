import torch
import numpy as np
import matplotlib.pyplot as plt


def inspect(x: torch.tensor, name: str = ''):
    print(f'tensor {name} data: {x.data} grad: {x.grad if x.requires_grad else None} requires_grad: {x.requires_grad}')


def show_grad_funs(xs: list[torch.tensor], names: list[str]):
    for x, name in zip(xs, names):
        print(f'{name}: {x.data} grad_fn: {x.grad_fn}')
    print()


def plot_learning(train_loss: np.ndarray, val_loss: np.ndarray, train_acc: np.ndarray, val_acc: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    xs = np.arange(len(train_loss))
    axes[0].plot(xs, train_loss, label='train loss')
    axes[0].plot(xs, val_loss, label='validation loss')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[1].plot(xs, train_acc, label='train acccuracy')
    axes[1].plot(xs, val_acc, label='train acccuracy')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
