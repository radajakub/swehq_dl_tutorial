import torch


def inspect(x: torch.tensor, name: str = ''):
    print(f'tensor {name} data: {x.data} grad: {x.grad if x.requires_grad else None} requires_grad: {x.requires_grad}')


def show_grad_funs(xs: list[torch.tensor], names: list[str]):
    for x, name in zip(xs, names):
        print(f'{name}: {x.data} grad_fn: {x.grad_fn}')
    print()
