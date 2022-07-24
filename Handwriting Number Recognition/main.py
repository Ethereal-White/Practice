import torch
import torch.nn.functional as F
from torchvision import transforms, datasets  # transforms可以对图像进行原始处理
from torch.utils.data import DataLoader
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def num_set(file_pathname):
    test_number = []
    for filename in os.listdir(file_pathname):
        img = cv2.imread(file_pathname+'/'+filename, cv2.IMREAD_GRAYSCALE)
        img = 1 - torch.Tensor(cv2.resize(img, (28, 28))) / 255
        img = img.view(1, 1, 28, 28)
        test_number.append(img)
    return test_number

num = num_set('test_number')

batch_size = 1
transform = transforms.Compose([
    transforms.ToTensor(),  # convert the PIL Image to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化 前者是均值 后者是标准差
])

test_dataset = datasets.MNIST(root='dataset/mnist', train=False,
                              download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

model = torch.load('model.pth')
a, b, c, d, e, f = model
w1 = torch.nn.Parameter(model[a])
b1 = torch.nn.Parameter(model[b])
w2 = torch.nn.Parameter(model[c])
b2 = torch.nn.Parameter(model[d])

def fc(x, weight, bias):
    weight = torch.transpose(weight, dim0=0, dim1=1)
    out = torch.matmul(x, weight) + bias
    return out

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv1.weight = w1
        self.conv1.bias = b1

        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2.weight = w2
        self.conv2.bias = b2

        self.pooling = torch.nn.MaxPool2d(2)
        # self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch, -1)
        return fc(x, model[e], model[f])

first = Net()
accuracy_list = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
first.to(device)

def test():  # This is just for test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            print(inputs.size())
            outputs = first(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total

    print('Accuracy on test set: %d %% [%d/%d]' % (accuracy, correct, total))
    accuracy_list.append(accuracy)

def recognize():
    for inputs in num:
        inputs = inputs.to(device)
        outputs = first(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        print(int(predicted[0]))

if __name__ == '__main__':
    recognize()
