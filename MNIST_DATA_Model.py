import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm




# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)
data_train_iter = iter(train_loader)
#images, labels = data_train_iter.next()
images, labels = next(data_train_iter)
print("Shape of the minibatch of images: {}".format(images.shape))
print("Shape of the minibatch of labels: {}".format(labels.shape))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2 = nn.Sequential(         
            nn.Conv2d(1,5, kernel_size=5,stride=3,padding=2),
            nn.ReLU(),
            #nn.Dropout2d(0.25)
        )
        self.lin = nn.Sequential(         
            nn.Linear(500, 10),
        )
    def forward(self, x):
        x = self.conv2(x)
        # flatten the output of conv2
        x = x.view(x.size(0), -1) 
        output = self.lin(x)
        # Apply softmax to x
        #output = F.log_softmax(x, dim=1)
        return output
    
model = CNN()
print(model)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Iterate through train set minibatchs 
for images, labels in tqdm(train_loader):
    # Zero out the gradients
    optimizer.zero_grad()
    
    # Forward pass
    #print(images.shape)
    #x = images.view(-1,28*28)
    y = model(images) 
    loss = criterion(y, labels)
    # Backward pass
    loss.backward()
    optimizer.step()
correct = 0
total = len(mnist_test)


with torch.no_grad():
    # Iterate through test set minibatchs 
    for images, labels in tqdm(test_loader):
        # Forward pass
        #x = images.view(-1, 28*28)
        y = model(images)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
    
print('Test accuracy: {}'.format(correct/total))