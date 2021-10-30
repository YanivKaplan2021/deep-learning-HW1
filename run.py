import torch
from torch.autograd import Variable

from networks import lenet5_network, lenet5_network_with_dropout, lenet5_network_with_batch_normaliztion
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
import matplotlib.pyplot as plt
import yaml, time

# Load configuration file
with open("config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_set = datasets.FashionMNIST("./data", download=True, transform=Compose([ToTensor()]))
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=Compose([ToTensor()]))

# Hyperparameters
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
use_gpu = config['use_gpu']
dropout = config['dropout']
weight_decay = config['weight_decay']
use_batch_normalization = config['use_batch_normalization']

# Arrangements
if dropout['use']:
    net = lenet5_network_with_dropout(dropout['dropout_ratio'])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
elif use_batch_normalization:
    net = lenet5_network_with_batch_normaliztion()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
elif weight_decay['use']:
    net = lenet5_network()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay['parameter'])
else:
    net = lenet5_network()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if use_gpu:
    net.cuda()
loss_fn = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

losses = []
for epoch in range(num_epochs):
    tic = time.perf_counter()
    # Train
    for i, (images, labels) in enumerate(train_loader):
        if use_gpu:
            images = images.float().cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.data)

    # Evaluation after each epoch
    train_accuracy = net.evaluation(train_loader, use_gpu)
    test_accuracy = net.evaluation(test_loader, use_gpu)
    toc = time.perf_counter()
    total_time = toc - tic
    print('Epoch : %d/%d,  Train Accuracy: %.4f, Test Accuracy: %.4f, Total Time: %d Seconds'
          % (epoch + 1, num_epochs, train_accuracy, test_accuracy, total_time))
