"""
Problem tanimi: CIFAR10 veri seti siniflandirma problemi
MNIST
CNN: Yapay Sinir Aglari 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% load dataset
def get_data_loader(batch_size = 64):
    
    transform = transforms.Compose([
        transforms.ToTensor(), # goruntuyu tensore cevir 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #RGB kanallarini normalize et
        ])
    
    #CIFAR10 veri setini indir ve egitim test kumelerini olustur.
    train_set = torchvision.datasets.CIFAR10("./data",train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    
    #DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle= False)
    
    return train_loader, test_loader


#%% data visualiation
def data_visualization(loader, n):
    images, labels = next(iter(loader))
    
    fig, axes = plt.subplots(1, n, figsize=(10,5))
    for i in range(n):
        axes[i].imshow(np.transpose(images[i],(1,2,0)))
        axes[i].set_title(f"Label {labels[i].item()}")
        axes[i].axis("off")
    plt.show()
    
# %% define cnn model

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        
        
        
    
    def forward(self, x):
        """
            # image (3x32x32) -> conv (32) -> relu (32) -> pool (16)
            # conv (16) -> relu (16) -> pool (8) -> image = 64x8x8
            flatten
            fc1 -> relu ->dropout
            fc2 -> output
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*8*8) # flatten islemi
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x



#define loss and optimizer

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ) 
        
        
        
#%% train
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            prediction = model(images)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader) 
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs},  AvgLoss:{avg_loss:.3f}")
    
    #loss graph
    plt.figure()
    plt.plot(range(1,epochs+1), train_losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train loss")
    plt.show()




#%% test

def test_model(model, test_loader, dataset_type):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            predictions = model(images)
            _, predictedIndices = torch.max(predictions, 1)
            correct += (predictedIndices == labels).sum().item()
            total += labels.size(0)
        
    print(f"{dataset_type} accuricy: {100*correct/total:.3f}%")



#%% main
if __name__ == "__main__":
    
    
    model = CNN().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_loader, test_loader = get_data_loader()
    data_visualization(train_loader, 5)
    #train_model(model, train_loader, criterion, optimizer)
    
    
    test_model(model, test_loader,"test")
    test_model(model, train_loader,"training")