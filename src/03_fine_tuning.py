# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torchvision import datasets, models, transforms
import numpy as np

from utils import plot_learning

folder = '../data/PACS_cartoon'
seed = 42
batch_size = 8

# %%
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %% in this task we will use a resnet18 architecture and weights
# we can download the weights directly from pytorch
net = models.resnet18(pretrained=True)
print(net)


# %% these are values I precomputed beforehand so we need not waste time
# normalization of the data is very important
mean = [0.8116, 0.7858, 0.7380]
std = [0.2730, 0.2863, 0.3374]

# %% load the data (lazily!!)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
train_data = datasets.ImageFolder(f'{folder}/train', transform)
test_data = datasets.ImageFolder(f'{folder}/test', transform)

# %% split the training data into validation and training sets
torch.manual_seed(seed)
np.random.seed(seed)

m = len(train_data)
shuffled_indices = np.random.permutation(m)
# put cca 20% of the data into the validation set
split_idx = int(m * 0.2)

# data loaders will ensure that we select the batches randomly
# they usually shuffle the data before every epoch
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                           sampler=torch.utils.data.SubsetRandomSampler(shuffled_indices[split_idx:]))
val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                         sampler=torch.utils.data.SubsetRandomSampler(shuffled_indices[:split_idx]))

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)


# %% function to evaluate the model on a given data
def evaluate(net, loader):
    correct = 0
    total_loss = 0
    N = 0

    # we do not need to compute gradients when evaluating the model
    with torch.no_grad():
        # for every batch in the data set (we cannot fit them into memory at once)
        for (x, t) in loader:
            N += x.size(0)

            x = x.to(dev)
            t = t.to(dev)

            scores = net(x)
            prediction = torch.argmax(scores, dim=1)

            correct += (prediction == t).sum()

            # takes log softmax over the last dimension
            probs = torch.log_softmax(scores, dim=-1)
            total_loss += F.nll_loss(probs, t, reduction='sum').item()

    accuracy = correct / N

    return accuracy, total_loss


# %% define a function for training the model
def train(net, train_loader, val_loader, lr=0.1, epochs=50):
    train_losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    val_accs = np.zeros(epochs)

    # we will skip crossvalidation since it is computationally expensive
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # train for full epochs
    for epoch in range(epochs):
        # go over the data once in each epoch
        for (x, t) in train_loader:
            # move the data to the device
            x = x.to(dev)
            t = t.to(dev)

            score = net(x)
            log_p = torch.log_softmax(score, -1)
            l = F.nll_loss(log_p, t, reduction='sum')

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            print(l.item())

        # evaluate both on training and validation sets
        train_accuracy, train_loss = evaluate(net, train_loader)
        val_accuracy, val_loss = evaluate(net, val_loader)

        print(f'epoch {epoch} train loss: {train_loss} train acc: {train_accuracy} val loss: {val_loss} val acc: {val_accuracy}')

        train_losses[epoch] = train_loss
        train_accs[epoch] = train_accuracy
        val_losses[epoch] = val_loss
        val_accs[epoch] = val_accuracy

    return train_losses, train_accs, val_losses, val_accs


# %% train the model from scratch
# ############################################3
full_net = models.resnet18(pretrained=False)

del full_net.fc

# replace the last layer of the network with a new classification layer (number of outputs as the number of classes)
linear_layer = nn.Linear(in_features=512, out_features=7)
nn.init.xavier_normal_(linear_layer.weight)
full_net.fc = linear_layer

# switch the network to train mode
full_net.train()
# move the network to the selected device
full_net.to(dev)

# %%
train_losses, train_accs, val_losses, val_accs = train(full_net, train_loader, val_loader, lr=0.003, epochs=50)

# %%
plot_learning(train_losses, val_losses, train_accs, val_accs)

# %%
test_acc, _ = evaluate(full_net, torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0))
print(test_acc)

# %% transfer-learning (learn only the new layer, freeze the remaining weights)
# ############################################3
transfer_net = models.resnet18(pretrained=True)

# freeze all weights
for param in transfer_net.parameters():
    param.requires_grad = False

# freeze batch-norm
transfer_net.eval()

del transfer_net.fc

# replace the last layer of the network with a new classification layer (number of outputs as the number of classes)
linear_layer = nn.Linear(in_features=512, out_features=7)
nn.init.xavier_normal_(linear_layer.weight)
transfer_net.fc = linear_layer

# move the network to the selected device
transfer_net.to(dev)

# %%
train_losses, train_accs, val_losses, val_accs = train(transfer_net, train_loader, val_loader, lr=0.001, epochs=50)

# %%
plot_learning(train_losses, val_losses, train_accs, val_accs)

# %%
test_acc, _ = evaluate(transfer_net, torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0))
print(test_acc)

# %% fine-tuning (learn all the weights but start from pre-trained)
# ############################################3
fine_net = models.resnet18(pretrained=True)

del fine_net.fc

# replace the last layer of the network with a new classification layer (number of outputs as the number of classes)
linear_layer = nn.Linear(in_features=512, out_features=7)
nn.init.xavier_normal_(linear_layer.weight)
fine_net.fc = linear_layer

# switch the network to train mode
fine_net.train()
# move the network to the selected device
fine_net.to(dev)

# %%
train_losses, train_accs, val_losses, val_accs = train(fine_net, train_loader, val_loader, lr=0.001, epochs=50)

# %%
plot_learning(train_losses, val_losses, train_accs, val_accs)

# %%
test_acc, _ = evaluate(fine_net, torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0))
print(test_acc)
