# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1o5A7eWPXNdB5gROQR8gcnGpCAhJMo0nA

### Typical performance

- **Random Guess**  
  F1-Score: 0.30  
  Accuracy: 0.50
- **Always Predict 1**  
  F1-Score: 0.37  
  Accuracy: 0.22
- **Always Predict 0**  
  F1-Score: 0.00  
  Accuracy: 0.77
- **sklearn.tree.DecisionTreeClassifier**  
  - **Training (5-fold cross-validation mean)**  
    F1-Score: 0.63-0.99  
    Accuracy: 0.85-0.99
  - **Validation (5-fold cross-validation mean)**  
    F1-Score: 0.50-0.60  
    Accuracy: 0.75-0.90
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score


def load_dataset(url):
  """ Get and load weather dataset. """

  path = url.split('/')[-1] # get the file name from url

  if not os.path.exists(path):
    print('Download:', url)
    urllib.request.urlretrieve(url, path)

  return pd.read_pickle(path) # pickle protocol=4


def get_input_target(df):
  """ Get X and y from weather dataset. """
  
  target_column = 'RainTomorrow' # predict 1 of it rains tomorrow

  X = df.drop(columns=[target_column]).to_numpy()
  y = df[target_column].to_numpy()

  return X, y


def k_fold_cv(model_create_fn, X, y, k=5):
  """ Run k-fold cross-validation. """

  results = []

  idxs = list(range(X.shape[0]))
  np.random.shuffle(idxs)

  for i, (train_idxs, val_idxs) in enumerate(KFold(k).split(idxs)):
    splits = {'train': (X[train_idxs], y[train_idxs]),
              'val':   (X[val_idxs],   y[val_idxs]  )}

    print('Run {}:'.format(i+1))

    model = model_create_fn()
    model.fit(*splits['train']) # training
    for name, (X_split, y_split) in splits.items():
      y_pred = model.predict(X_split)
      result = {'split': name,
                'f1': f1_score(y_pred, y_split), #(prdict結果，正確結果)
                'acc': accuracy_score(y_pred, y_split)}
      results.append(result)
      print('{split:>8s}: f1={f1:.4f} acc={acc:.4f}'.format(**result))

  return pd.DataFrame(results)

import copy

class Node:
  def __init__(self,feature=None,value=None,Left=None,Right=None,gini=None,guess=None,depth=None):
    self.feature=feature
    self.value=value
    self.Left=Left
    self.Right=Right
    self.guess=guess
    self.gini=gini
    self.depth=depth


def SplitDataSet(dataSet, feature, value):

    ge_thresh = dataSet[:,feature] > value
    
    left = dataSet[~ge_thresh]
    right = dataSet[ge_thresh]
    
    return left, right

#caculate the gini value
def gini(dataSet):
  #answer=>final column
  answer=len(dataSet[0])-1
  uniquefeature=np.unique(dataSet[:,answer])
  f=[]
  for feature in uniquefeature:
    #append the feature count in a count list
    f.append(np.count_nonzero(dataSet[:,answer] == feature))
  gini=1
  for count in f:
    gini-=(float(count/sum(f)))**2
  return gini


def chosefeature(dataSet):
  chose={}
  for feature in range(len(dataSet[0])-2):
    dataSet = dataSet[dataSet[:,feature].argsort()]
    unique=np.unique(dataSet[:,feature])
    mingini={}
    if len(unique)==1:
      continue
    for i in range(len(unique)-1):
      cut=unique[i]
      left=dataSet[dataSet[:,feature] <= cut]
      right=dataSet[dataSet[:,feature] > cut]
      total=len(left)+len(right)
      gi=((len(left)/total)*gini(left))+((len(right)/total)*gini(right))
      mingini[cut]=gi
    cutvalue=min(mingini, key=lambda k: mingini[k])#arg
    chose[feature]=[cutvalue,mingini[cutvalue]]
  if len(chose)==0:
    return -1,-1
  chosefeature=min(chose, key=lambda k: chose[k][1])#arg
  cutvalue=chose[chosefeature][0]
  return chosefeature,cutvalue

def guess(dataset):
  f={}
  uniquefeature=np.unique(dataset[:,len(dataset[0])-1])
  for feature in uniquefeature:
    #append the feature count in a count list
    f[feature]=np.count_nonzero(dataset[:,len(dataset[0])-1] == feature)
  guess= max(f.keys(), key=(lambda k: f[k]))
  return guess 


def decisiontree(now,data):
  num_col=len(data[0])
  global leaf
  # all the data in the set is all in the same category
  if len(np.unique(data[:,num_col-1]))==1:
    return
  feature,cut=chosefeature(data)
  if cut==-1:
    return
  Leftdata,Rightdata=SplitDataSet(data, feature, cut)
  if now.guess==None:
    now.guess=guess(data)
  Leftguess=guess(Leftdata)
  Rightguess=guess(Rightdata)
  nowp=np.full(len(data), now.guess)
  leftp=np.full(len(Leftdata), Leftguess)
  rightp=np.full(len(Rightdata), Rightguess)
  #Check if the f1_score will become higher after the split
  Nowacc=f1_score(nowp,data[:,num_col-1])
  if len(np.unique(Leftdata[:,num_col-1]))==1 or len(np.unique(Rightdata[:,num_col-1]))==1:
    AfterCutacc=1
  else:
    AfterCutacc=(f1_score(leftp,Leftdata[:,num_col-1])*len(Leftdata)/len(data))+(f1_score(rightp,Rightdata[:,num_col-1])*len(Rightdata)/len(data))
  if AfterCutacc<Nowacc:
    return
  Left=Node(depth=now.depth+1,guess=Leftguess)
  Right=Node(depth=now.depth+1,guess=Rightguess)
  #Update the node
  now.Left=Left
  now.Right=Right
  now.feature=feature
  now.value=cut
  return decisiontree(Left,Leftdata),decisiontree(Right,Rightdata)


def classify(tree,data):
  if tree.value==None:
    return tree.guess
  if data[tree.feature]<=tree.value:
    return classify(tree.Left,data)
  else:
    return classify(tree.Right,data)
    
#Decision Tree Model
class Model:

  def __init__(self, num_features: int, num_classes: int):

    self.num_features = num_features
    self.num_classes = num_classes
    self.trees=[]
    self.k=8 #8 trees
    self.len=4 # how big is the tree


  #Build the DecisionTree by the Training Data
  def fit(self, X: np.ndarray, y: np.ndarray):

    #Combine X & y for convenience
    y=y.reshape(y.shape[0],1)
    Xy=np.append(X,y, axis=1)

    #For k Decision Tree
    for i in range(self.k):
      Tree=Node(feature=0,depth=0)
      sample = Xy[np.random.choice(Xy.shape[0], size=int(len(Xy)/self.len), replace=True)]
      decisiontree(Tree,sample)
      self.trees.append(Tree)
    

  def predict(self, X: np.ndarray) -> np.ndarray:
    '''
    Predict y given X.

    Args:
        X (np.ndarray) : inputs, shape: (num_inputs, num_features).
    
    Returns:
        np.ndarray : the predicted integer outputs, shape: (num_inputs,).
    '''

    p=[]
    for i in range(len(X)):
      guess=[]
      #the guess predicted by the forest
      for j in range(self.k):
        guess.append(classify(self.trees[j],X[i]))
      #Majority rule
      f={}
      uniquefeature=np.unique(guess)
      for feature in uniquefeature:
        #append the feature count in a count list
        f[feature]=np.count_nonzero(guess == feature)
      p.append(max(f.keys(), key=(lambda k: f[k])))
    p=np.array(p)
    return p



#load the data
df = load_dataset('https://lab.djosix.com/weather.pkl')
X_train, y_train = get_input_target(df)

df.head()

create_model = lambda: Model(X_train.shape[1], 2)
k_fold_cv(create_model, X_train, y_train).groupby('split').mean()