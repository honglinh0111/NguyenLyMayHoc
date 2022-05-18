import pandas as pd
import numpy as np
from scipy.io import arff
from pprint import pprint
dataset=pd.read_csv('csv_result-messidor_features.csv')

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
dataset = shuffle(dataset)
dataset.reset_index (inplace = True, drop = True)
#print(dataset)
X=dataset.iloc[:,1:19]
y=dataset.iloc[:,19:20]
#print(X)
#print(y)
print ('Dem so lan xuat hien cua 2 gia tri 0 va 1 của nhan Class')

