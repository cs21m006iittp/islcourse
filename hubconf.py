import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )



    
def create_dataloaders(training_data, test_data, batch_size=64):

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
        
    return train_dataloader, test_dataloader

class ModifiedDataset(Dataset):
  def __init__(self,given_dataset,shrink_percent=10):
    self.given_dataset = given_dataset
    self.shrink_percent = shrink_percent
    
  def __len__(self):
    return len(self.given_dataset)

  def __getitem__(self,idx):
    img, lab = self.given_dataset[idx]

    # print (type(img))
    # print (img.shape)

    img2 = transform_tensor_to_pil(img.squeeze())

    # print (img2.size)
    
    new_w = int(img2.size[0]*(1-self.shrink_percent/100.0))
    new_h = int(img2.size[1]*(1-self.shrink_percent/100.0))

    # print (new_w, new_h)

    img3 = img2.resize((new_w,new_h))

    # print (img3.size)

    x = transform_pil_to_tensor(img3)

    # print (x.shape)

    return x,lab





class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

print(net)



loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


training_data, test_data = load_data()

train_dataloader, test_dataloader = create_dataloaders(training_data, test_data)
input, label = trainingdata

def test_model(dataloader, net, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    net.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = net(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
 def train_model(dataloader, net, loss_fn, optimizer):
    size = len(dataloader.dataset)
    net.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = net(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
            


epochs = 10 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_model(train_dataloader, net, loss_fn, optimizer)
    test_model(test_dataloader, net, loss_fn)
print("Done!")


