# %% imports
import torch
import torch.nn as nn

import numpy as np

from data_model import DataModel
from utils import plot_learning

# %% define parameters for the task
train_size = 200
# our advantage that we can generate as much points as we want
# in reality we would have one dataset and would need to split it to three sets
val_size = 1000
test_size = 30000


# %% initialize data model and sample datasets
data_model = DataModel()
x, t = data_model.generate_sample(train_size)
xv, tv = data_model.generate_sample(val_size)
xt, tt = data_model.generate_sample(test_size)

print(f'train points: {x.shape}')
print(f'train labels: {x.shape}')


# %% visualize training data
data_model.plot_boundary((x, t), title='Training data')


# %% build the neural network

# we face a binary classification problem
# we will build a neural network to perform logistic regression on lifted points to find a separating boundary
# assume that the input size is 2 since we have 2D points and the size of the ouptut is 1 since for every point we want to
class MyNet(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        # implement ReLU(W_1 * x + b_1) * w + b in a network
        # then try to replace ReLU with Tanh

        self.layers = nn.Sequential(
            # modules take care of setting the requires grad for the parameters
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.apply(self.init_layers)

    def forward(self, x: torch.tensor):
        return self.layers(x).squeeze(1)

    def classify(self, x: torch.tensor):
        return torch.sign(self.forward(x))

    def nll_loss(self, scores: torch.tensor, t: torch.tensor):
        # - 1/N * sum(log(sigmoid(t * scores)))
        return -torch.mean(torch.log(torch.sigmoid(t * scores)))

    def accuracy(self, x: torch.tensor, t: torch.tensor):
        with torch.no_grad():
            y = self.classify(x)
            return (y == t).float().mean()

    def init_layers(self, layer: nn.Module):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -1.0, 1.0)
            nn.init.uniform_(layer.bias, -1.0, 1.0)


# %% now do the training loop where we will minimize the negative log likelihood
# try hidden sizes 2, 5, 10, 100, 500
hidden_size = 2
epochs = 1000

net = MyNet(hidden_size=hidden_size)

# optimizer helps us with applying the gradient steps
# we do not have to think about zeroing the gradients etc.
# moreover, some optimizers have additional features like momentum, weight decay, etc.
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

training_losses = np.zeros(epochs)
training_accs = np.zeros(epochs)
validation_losses = np.zeros(epochs)
validation_accs = np.zeros(epochs)

# becase our dataset is small we can perform the full gradient update
for epoch in range(epochs):
    # forward pass through the network
    # we have a score which is in the range of (-inf, inf)
    scores = net(x)

    # compute the negative log likelihood loss
    l = net.nll_loss(scores, t)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    training_losses[epoch] = l.item()
    training_accs[epoch] = net.accuracy(x, t).item()
    with torch.no_grad():
        validation_losses[epoch] = net.nll_loss(net(xv), tv).item()
    validation_accs[epoch] = net.accuracy(xv, tv).item()

    # compute classification accuracy of the model
    print(f'epoch {epoch} training loss: {l.item()} training acc: {training_accs[epoch]}')

# %% visualize the training loss
plot_learning(training_losses, validation_losses, training_accs, validation_accs)

# %% visualize the decision boundary
data_model.plot_boundary((x, t), predictor=net)

# %% evaluate the test error
print(f'test accuracy: {net.accuracy(xt, tt).item()}')

# %%
# validation is needed when we have some hyperparameters and we want to find the best values for them
# also we should not use validation test for training, it serves as a proxy for generalization
# we CANNOT use the test set for validation, it must not be used in learning in any way
# we typically get only test and training sets so we need to split the training set to training and validation
