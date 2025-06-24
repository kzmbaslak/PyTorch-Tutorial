# -*- coding: utf-8 -*-
"""
Problem tanimi: mnist veri seti ile rakam siniflandirma projesi
MNIST
ANN: Yapay Sinir Aglari 
"""

#%% library

import torch
import torch.nn as nn #yapay sinir agi katmanlarini tanimlamak icin kullan
import torch.optim as optim #optimizasyon algoritmalarini iceren modul
import torchvision # goruntu isleme ve pre-defined(trained) modelleri icerir
import torchvision.transforms as transforms # goruntu donusumleri yapmak
import matplotlib.pyplot as plt 

#otional: cihaz belirleme
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data loading
def get_data_loaders(batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),#goruntuyu tensore cevirir 0-255 -> 0-1 olceklenir
        transforms.Normalize((0.5,), (0.5,)), #piksel degerlerini -1-1 arasına olcekler.
        ])
    #mnist veri setini indir ve egitim test kumelerini olustur.
    train_set = torchvision.datasets.MNIST("./data",train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    #pytorch veri yukleyicisi olustur
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    
    return train_loader, test_loader
    
#train_loader, test_loader = get_data_loaders()
# data visualization
def visualize_semples(loader,n):
    images,labels = next(iter(loader)) #ilk batch goruntu ve etiketleri alir
    #print(images[0].shape)
    fig, axes = plt.subplots(1,n,figsize=(10,5)) # n farkli gorutuyu gorsellestirme alani
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap = "gray") 
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off") #eksenleri gizle
    plt.show()

#visualize_semples(train_loader, 5)
# %% define ann model

#yapay sinir agi calss
class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.fletten = nn.Flatten() # 2D gorsellerimizi vektor haline cevirelim(1D) -> 28*28 = 784
        
        self.fc1 = nn.Linear(28*28, 128) # ilk tam bagli katmani olustur.
        
        self.relu = nn.ReLU()# aktivasyon fonksiyonu
        
        self.fc2 = nn.Linear(128, 64)# ikinci tam bagli katmani olustur.
        
        self.fc3 = nn.Linear(64, 10)# cikti katmani olustur.

    def forward(self, x): # forward propagation x = goruntu
        x = self.fletten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
#create model and compile
#model = NeuralNetwork().to(DEVICE)

# kayip fonksiyonu ve optimizasyon algoritmasini belirle
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.Adam(model.parameters(), lr= 0.0001)
    )

#criterion, optimizer = define_loss_and_optimizer()

# %% train
def train_model(model, train_loader, criterion, optimizer, epochs = 10):
    
    model.train() # modeli egitim moduna al
    train_losses = [] # her bir epoch sonucundaki loss degerlerini saklamak icin liste
    
    for epoch in range(epochs):
        total_loss = 0 # toplam kayip degeri
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
    
            optimizer.zero_grad() # gradyanlari sifirla
            predictions = model(images) # modeli uygula forward prop.
            loss = criterion(predictions, labels) # loss hesapla
            loss.backward() # geri yayilim  yani gradyan hesaplama
            optimizer.step() # update weights (agirliklari guncelle)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")
        
    # loss graph
    plt.figure()
    plt.plot(range(1,epochs + 1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
    
#train_model(model, train_loader, criterion, optimizer, epochs = 5)            

#%% test
def test_mmodel(model, test_loader):
    model.eval() # modeli degerlendirme moduna al
    correct = 0 # toplam dogru tahmin sayaci
    total = 0 # toplam veri sayaci
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)
            
            _, predicted = torch.max(predictions, 1) # maximum 1 adet deger sec
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"{100*correct/total:.3f}%")

#test_mmodel(model, test_loader)
            
            

#%% main

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_semples(train_loader, 5)
    model = NeuralNetwork().to(DEVICE)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer)
    test_mmodel(model, test_loader)
    