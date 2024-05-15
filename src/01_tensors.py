# %% imports
import torch

# ignore warnings just to show some possible errors in torch without the console reporting it
import warnings
warnings.filterwarnings("ignore")


# %% functions to display information about tensors
def inspect(x: torch.tensor, name: str = ''):
    print(f'tensor {name} data: {x.data} grad: {x.grad if x.requires_grad else None} requires_grad: {x.requires_grad}')


def show_grad_funs(xs: list[torch.tensor], names: list[str]):
    for x, name in zip(xs, names):
        print(f'{name}: {x.data} grad_fn: {x.grad_fn}')
    print()


# %% define tensors and their initial values similarly to the presentation

# input values
a = torch.tensor(1.0)
b = torch.tensor(2.0)

# parameters
w = torch.tensor(2.0)
t = torch.tensor(4.0)


# %% inspect a
inspect(a, name='a')
inspect(b, name='b')
inspect(w, name='w')
inspect(t, name='t')


# %% define function on the tensors L = ((a + b) * w - t) ** 2 + w ** 2
# by composing the operations on the tensors torch builds computational graph
def forward(a: torch.tensor, b: torch.tensor, w: torch.tensor, t: torch.tensor, show: bool = False) -> torch.tensor:
    x1 = a + b
    x2 = x1 * w
    x3 = x2 - t
    x4 = x3 ** 2  # or x3.pow(2) or torch.pow(x3, 2)
    x5 = w ** 2
    L = x4 + x5

    if show:
        show_grad_funs([x1, x2, x3, x4, x5, L], ['x1', 'x2', 'x3', 'x4', 'x5', 'L'])

    return L


# %% apply the function to the tensors
L = forward(a, b, w, t)

inspect(a, name='a')
inspect(b, name='b')
inspect(w, name='w')
inspect(t, name='t')

# notice that no tensor has requires_grad=True, so no gradients will be computed in the backward pass and no graph was built
# torch is smart and does not compute the graph when there is no parameter that needs the gradient

# %% try to compute the gradients -> throws error
L.backward()


# %% set gradients required for w and t
w.requires_grad = True
t.requires_grad = True


# %% compute L again and try backward pass again
L = forward(a, b, w, t, show=True)
inspect(a, name='a')
inspect(b, name='b')
inspect(w, name='w')
inspect(t, name='t')


# %% now try to call backward again and see the gradients in the vectors
L.backward()
inspect(a, 'a')
inspect(b, 'b')
inspect(w, 'w')
inspect(t, 't')


# %% try to apply backward again (the gradient is not retained in non-leaf tensors by default)
L.backward()

# %% apply forward and backward again to see that the gradients accumulate and we need to manually reset them
L = forward(a, b, w, t)
L.backward()
inspect(a, 'a')
inspect(b, 'b')
inspect(w, 'w')
inspect(t, 't')

# %% reset the gradients before every backward pass
w.grad = None
t.grad = None

L = forward(a, b, w, t)
L.backward()
inspect(a, 'a')
inspect(b, 'b')
inspect(w, 'w')
inspect(t, 't')

# %% make gradient step and inspect the tensors (use with torch.no_grad() to avoid storing the computation graph)
lr = 0.1

w1 = w - lr * w.grad
t1 = t - lr * t.grad

# %% inspect the tensors and their grad_function
show_grad_funs([w1, t1], ['w1', 't1'])

# %% that is why we need to prevent the computation graph from being stored
with torch.no_grad():
    w2 = w - lr * w.grad
    t2 = t - lr * t.grad

show_grad_funs([w2, t2], ['w2', 't2'])

# %% if we want to do it in-place we can use the .data attribute
with torch.no_grad():
    w.data -= lr * w.grad
    t.data -= lr * t.grad

show_grad_funs([w, t], ['w', 't'])
