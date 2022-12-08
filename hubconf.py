
# kali
import torch
from torch import nn
import torch.optim as optim
from sklearn.cluster import KMeans  
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.datasets import load_digits 
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
import torch


from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor
import torch.nn.functional as F

# imporitn the data set from the sklearn

#import the classifier and performance matrix

from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_blobs(n_samples=n_points,centers = 5)
  print(X.shape, y.shape)
  # write your code ...
  return X,y

def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_circles(n_samples=n_points, noise=0.05)
  # write your code ...
  return X,y

def get_data_mnist():
  pass
  # write your code here
  # Refer to sklearn data sets
  digits = load_digits()
  X , y = digits.data, digits.target
  print(X.shape, y.shape)
  # write your code ...
  return X,y

def build_kmeans(X,k=10):
  pass
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  
  km = KMeans(n_clusters=k)
  km.fit(X)
  # write your code ...
  return km

def assign_kmeans(km,X):
  pass
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = 0,0,0 # you need to write your code to find proper values
  v = v_measure_score(ypred_1,ypred_2)
  h = homogeneity_score(ypred_1,ypred_2)
  c = completeness_score(ypred_1,ypred_2)
  return h,c,v

###### PART 2 ######

def build_lr_model(X=None, y=None):
  pass
  lr_model = LogisticRegression(random_state=0).fit(X, y)
  # write your code...
  # Build logistic regression, refer to sklearn
  return lr_model

def build_rf_model(X=None, y=None):
  pass
  rf_model = RandomForestClassifier(n_estimators=100)
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  rf_model.fit(X,y)
  return rf_model

def get_metrics(model=None,X=None,y=None):
  pass
  ypred = model.predict(X)
  print("Classification Report is : ")
  print(classification_report(y,ypred))
  print("Confusion Matrix is : ")
  print(confusion_matrix(y, ypred))
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  print("f1 score is : ",f1_score(y, ypred, average=None))
  print("Precision score is :",precision_score(y, ypred, average=None))
  print("recall score is : ",recall_score(y, ypred, average=None))
  print("accuracy score is : ", accuracy_score(y, ypred))
  acc = accuracy_score(y, y_pred)
  
  prec = precision_score(y, ypred, average=None)
  rec = recall_score(y, ypred, average=None)
  f1 = f1_score(y, ypred, average=None)
  auc = roc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr')
  
  # write your code here...
  return acc, prec, rec, f1, auc


def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid =  {
    'n_estimators' : [50, 100],
    'max_features' : ['auto', 'sqrt','log2'],
    'max_depth' : [0, 1, 10],
    'criterion' : ['gini', 'entropy']
}

  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
  grid_search_cv = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv)
  # create a grid search cv object
  # fit the object on X and y input above
  new_model = grid_search_cv.fit(X,y)
  # write your code here...
  ypred =new_model.predict(X)
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  acc = accuracy_score(y, ypred)
  
  prec = precision_score(y, ypred, average=None)
  rec = recall_score(y, ypred, average=None)
  f1 = f1_score(y, ypred, average=None)
  roc = roc_auc_score(y, new_model.predict_proba(X), multi_class='ovr')
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  a = acc.max()
  b = prec.max() 
  c =  f1.max()
  d =rec.max()
  e = roc.max()
  top1_scores = [a,b,c,d,e]
  
  return top1_scores

###### PART 3 ######

class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self)
    
    self.fc_encoder = nn.Linear(in_features = inp_dim, out_features = hid_dim)# write your code inp_dim to hid_dim mapper
    self.fc_decoder = nn.Linear(in_features = hid_dim, out_features = inp_dim) # write your code hid_dim to inp_dim mapper
    self.fc_classifier = nn.Linear(in_features = hid_dim, out_features = num_classes) # write your code to map hid_dim to num_classes
    
    self.relu = nn.ReLu()#write your code - relu object
    self.softmax = nn.Softmax() #write your code - softmax object
    
  def forward(self,x):
    x = nn.Flatten(x) # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    e = 0.001
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    lc1 = -torch.sum(yground*torch.log(y_pred+e))# write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  digits = load_digits()
  X , y =  tf.convert_to_tensor(digits.data) , tf.convert_to_tensor(digits.target)
  print(X.shape, y.shape)
  # write your code ...
  # convert to tensor
  # write your code
  return X,y

def get_loss_on_single_point(mynn=None,x0,y0):
  y_pred, xencdec = mynn(x0)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  # the lossval should have grad_fn attribute set
  return lossval

def train_combined_encdec_predictor(mynn=None,X,y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    optimizer.zero_grad()
    ypred, Xencdec = mynn(X)
    lval = mynn.loss_fn(X,y,ypred,Xencdec)
    lval.backward()
    optimzer.step()
    
  return mynn
    




  
  
  
  
