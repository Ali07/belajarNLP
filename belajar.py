#data analysis library
import numpy as np
import pandas as pd

#visualization library
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#ingore warning
import warnings
warnings.filterwarnings('ignore')

#import dataset

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

#look at training data
train.describe(include="all")