import torch
import torch.nn.functional as F
from torchvision import transforms, datasets  # transforms可以对图像进行原始处理
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # convert the PIL Image to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化 前者是均值 后者是标准差
])

train_dataset = datasets.MNIST(root='dataset/mnist', train=True,
                               download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='dataset/mnist', train=False,
                              download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        return self.fc(x)


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 0表示第一块显卡
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

accuracy_list = []


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if epoch % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total
    print('Accuracy on test set: %d %% [%d/%d]' % (accuracy, correct, total))
    accuracy_list.append(accuracy)


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
    save_path = ''
    torch.save(model.state_dict(), 'model.pth')
    plt.plot(range(10), accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
