import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

lr = 1e-4
dropout_rate = 0
weight_decay = 0.01

transform = transforms.Compose([
    transforms.ToTensor()
    ])
train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

X_train = [train_data[i][0] for i in range(len(train_data))]
Y_train = [train_data[i][1] for i in range(len(train_data))]

X_test = [test_data[i][0] for i in range(len(test_data))]
Y_test = [test_data[i][1] for i in range(len(test_data))]

# Class number to name
num2class = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# Sample data
sample_idx = torch.randint(len(X_train), size=(1,)).item()
img, label = X_train[sample_idx], Y_train[sample_idx]

#Look at data
plot_sample = False
if plot_sample:
    fig, ax = plt.subplots()
    plt.title(num2class[label])
    plt.imshow(img.squeeze())
    plt.show()


class LeNet5(nn.Module):
    def __init__(self, dropout_rate=0, BN=False):
        super(LeNet5, self).__init__()
        self.BN = BN
        self.activation = nn.Tanh()
        self.avg_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.BN1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.BN2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.BN3 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
        self.final_activation = nn.Softmax()

    def forward(self, x):
        x = self.activation(self.conv1(x)) if self.BN is False else self.activation(self.BN1(self.conv1(x)))
        x = self.activation(self.avg_pooling(x))
        x = self.activation(self.conv2(x)) if self.BN is False else self.activation(self.BN2(self.conv2(x)))
        x = self.activation(self.avg_pooling(x))
        x = self.activation(self.conv3(x)) if self.BN is False else self.activation(self.BN3(self.conv3(x)))
        x = x.view(x.size(0))
        x = self.activation(self.fc1(x))
        x = self.final_activation(self.fc2(x))
        return x


model = LeNet5()
y_probs = model(img, dropout_rate=dropout_rate, BN=False)
y_hat = torch.argmax(y_probs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)







