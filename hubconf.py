import torchmetrics
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchmetrics import Precision, Recall, Accuracy, F1Score
from torchmetrics.classification import accuracy


#Selecting the Divice----------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

#Function for Loading the data-------------------------------------
def load_data():
  training_data = datasets.FashionMNIST( 
      root = "data",
      train=True,
      download=True,
      transform=ToTensor(),
  ) 

  test_data = datasets.FashionMNIST(
      root= "data",
      train = False,
      download=True,
      transform = ToTensor(),
  )
  return training_data, test_data




#creating Data Loader==========================================================
def Create_loader(training_data, test_data, batch_size):
  train_loader = DataLoader(training_data,batch_size,shuffle=True)
  test_loader = DataLoader(test_data,batch_size,shuffle=True)

  for X, Y in train_loader:
    print(f"Shape of the input data X is : {X.shape}")
    print(f"Shape of the output data Y is : {Y.shape}")
    break
  
  return train_loader, test_loader



#defining loss function=========================================================
def loss_fn(Y_pred, Y_ground):
  e = 0.001
  v = -torch.sum(Y_ground*torch.log(Y_pred+e))
  return v


# Creating NN ====================================================================

class CS21M006(nn.Module):
  def __init__(self, channels, classess, hight, width, config):
    super(CS21M006, self).__init__()

    self.seq_op = nn.Sequential()
    self.param = 0

    for i in range(len(config)):
      layer = config[i]

      in_channel = layer[0]
      out_channel = layer[1]
      kernel = layer[2]
      padding = layer[4]

      if isinstance(layer[3], int):
        stride = [layer[3], layer[3]]
      else:
        stride = layer[3]


      self.seq_op.append(nn.Conv2d(in_channels= in_channel, out_channels = out_channel, kernel_size = kernel, stride = stride, padding = padding ).to(device))

      self.seq_op.append(nn.ReLU())

      if padding != "same":
        if isinstance(padding, int ):
          padding = [padding, padding]
        hight = (int)((hight-kernel[0])/stride[0])+1
        width = (int)((width- kernel[1])/stride[1])+1

      if i == len(config)-1:
        self.param = out_channel
    self.seq_op.append(nn.Flatten())
    self.seq_op.append(nn.Linear(self.param*hight*width,classess).to(device))
    self.seq_op.append(nn.Softmax(1))


  def forward(self, x):
    x= self.seq_op(x)
    return x

# function for getting loss function ====================================================
def get_lossfn_and_optimizer():
    lossfn = loss_fn
    
    return lossfn     

#function for training the model for each epoch======================================================    
def train_model(train_loader,model,loss_fn,optimizer):
  size = len(train_loader.dataset)
  batches = len(train_loader)
  classes = len(train_loader.dataset.classes)
  model.train()
  t_loss = 0
  for batch, (X, Y) in enumerate(train_loader):
    X, Y = X.to(device), Y.to(device)
    #predictating the value
    pred = model(X.to(device))
    Y= F.one_hot(Y.to(device),10)
    #computing the error
    loss = loss_fn(pred, Y)
    t_loss += loss.item()
    #optimizing the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f" Avg Loss : {loss/batches:>7f} \n")



#train function for calling the train_model function for each epocch
def train(train_loader, model, epochs ,learning_rate=1e-3):
  loss_f = loss_fn
  optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum =0.9)
  for i in range(epochs):
        print("running epoch ",i)
        train_model(train_loader,model,loss_f,optimizer)
  print("FINISHED TRAINING:")


#Test function for testting the model
def test(test_loader,model, loss_fn):
  size = len(test_loader.dataset)
  batches = len(test_loader)
  classes = len(test_loader.dataset.classes)

  model.eval()
  Y_pred = []
  Y_true = []
  loss , correct = 0 ,0
  with torch.no_grad():
    for X ,Y in test_loader:
      pred = model(X.to(device))

      t = F.one_hot(Y.to(device), 10).to(device)
      Y_pred.append(pred.argmax(1))
      Y_true.append(Y)
      loss += loss_fn(pred, t).item()
      correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
  
  loss /= batches
  correct /= size
  Y_pred = torch.cat(Y_pred)
  Y_true = torch.cat(Y_true)
  print(f"\nAvg loss: {loss:>8f} \n")
  

  accuracy = Accuracy().to(device)
  print(f'Accuracy : {accuracy(Y_pred, Y_true).item()*100:.3f}% ')

  precision = Precision(average = 'macro', num_classes = classes).to(device)
  print(f'precision :{ precision(Y_pred,Y_true).item():.4f}')

  recall = Recall(average = 'macro', num_classes = classes).to(device)
  print(f'recall : { recall(Y_pred,Y_true).item():.4f}')
  f1_score = F1Score(average = 'macro', num_classes = classes).to(device)
  print(f'f1_score :  {f1_score(Y_pred,Y_true).item():.4f}')
  return accuracy,precision, recall, f1_score





#Get_model function for gettting and trained on the given configuration

def get_model(trainloader,config,epochs,learning_rate):
    
    N , num_channels , height , width = next(iter(trainloader))[0].shape
    num_classes = len(trainloader.dataset.classes)
    
    my_model = CS21M006(num_channels,num_classes,height,width,config).to(device)
    
    train(trainloader,my_model,epochs,learning_rate)
    
    return my_model

