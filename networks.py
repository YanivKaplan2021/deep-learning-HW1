import torch
from torch import nn
import torch.nn.functional as functional

class lenet5_network(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

    self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)

  def forward(self, n):
    # Convolutional layer 1
    n = self.conv1(n)
    n = functional.relu(n)
    n = functional.max_pool2d(n, kernel_size=2, stride=2)

    # Convolutional layer 2
    n = self.conv2(n)
    n = functional.relu(n)
    n = functional.max_pool2d(n, kernel_size=2, stride=2)

    # flattening before inner production layer
    input_to_linear = 16*4*4
    # fc1
    n = n.reshape(-1, input_to_linear)
    n = self.fc1(n)
    n = functional.relu(n)

    # fc2
    n = self.fc2(n)
    n = functional.relu(n)

    # output
    n = self.out(n)
    # We'll apply cross-entropy as loss function later

    return n

  def evaluation(self, dataloader, use_gpu):
    total, correct = 0, 0
    # keeping the network in evaluation mode
    self.eval()
    with torch.no_grad():
      for data in dataloader:
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = self(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    # returning the network to train mode
    self.train()
    return 100 * correct / total

class lenet5_network_with_dropout(nn.Module):
  def __init__(self, dropout_ratio):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

    self.dropout_ratio = dropout_ratio

    self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)

  def forward(self, n):
    # Convolutional layer 1
    n = self.conv1(n)
    n = functional.dropout(n, self.dropout_ratio)
    n = functional.relu(n)
    n = functional.max_pool2d(n, kernel_size=2, stride=2)

    # Convolutional layer 2
    n = self.conv2(n)
    n = functional.dropout(n, self.dropout_ratio)
    n = functional.relu(n)
    n = functional.max_pool2d(n, kernel_size=2, stride=2)

    # flattening before inner production layer
    input_to_linear = 16*4*4
    # fc1
    n = n.reshape(-1, input_to_linear)
    n = self.fc1(n)
    n = functional.relu(n)

    # fc2
    n = self.fc2(n)
    n = functional.relu(n)

    # output
    n = self.out(n)
    # We'll apply cross-entropy as loss function later

    return n

  def evaluation(self, dataloader, use_gpu):
    total, correct = 0, 0
    # keeping the network in evaluation mode
    self.eval()
    with torch.no_grad():
      for data in dataloader:
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = self(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    # returning the network to train mode
    self.train()
    return 100 * correct / total

class lenet5_network_with_batch_normaliztion(nn.Module):
  def __init__(self):
    super().__init__()

    # convolutional layers
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    self.conv1_bn = nn.BatchNorm2d(6)

    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
    self.conv2_bn = nn.BatchNorm2d(16)

    self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)

  def forward(self, n):
    # Convolutional layer 1
    n = self.conv1(n)
    n = self.conv1_bn(n)
    n = functional.relu(n)
    n = functional.max_pool2d(n, kernel_size=2, stride=2)

    # Convolutional layer 2
    n = self.conv2(n)
    n = self.conv2_bn(n)
    n = functional.relu(n)
    n = functional.max_pool2d(n, kernel_size=2, stride=2)

    # flattening before inner production layer
    input_to_linear = 16*4*4
    # fc1
    n = n.reshape(-1, input_to_linear)
    n = self.fc1(n)
    n = functional.relu(n)

    # fc2
    n = self.fc2(n)
    n = functional.relu(n)

    # output
    n = self.out(n)
    # We'll apply cross-entropy as loss function later

    return n

  def evaluation(self, dataloader, use_gpu):
    total, correct = 0, 0
    # keeping the network in evaluation mode
    self.eval()
    with torch.no_grad():
      for data in dataloader:
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = self(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    # returning the network to train mode
    self.train()
    return 100 * correct / total
