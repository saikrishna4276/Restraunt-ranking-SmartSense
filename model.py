import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegressionCV
import sklearn.metrics as metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import json
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
from IPython.display import display, Math, Latex
from surprise import SVD,BaselineOnly, Reader,KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', 450)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")

c0=sns.color_palette()[0]
c1=sns.color_palette()[1]
c2=sns.color_palette()[2]

import json

def readjson(filepath):
    data = []
    i=0
    with open(filepath,encoding="utf8") as f:
            for line in f:
                data.append(json.loads(line))
                i +=1
    return pd.DataFrame(data)

data = readjson('./yelp_academic_dataset_business.json')

data['categories'] = data['categories'].astype(str)
restaurant_df = data[data['categories'].str.contains('Food')==True]

user_id = []
for i in range(1,len(restaurant_df)+1):
  user_id.append(i)
restaurant_df['user_id'] = user_id

baseline_df = restaurant_df[['user_id','business_id','stars']]

reader = Reader(rating_scale=(1, 5))
# Load the dataset
# and split it into 3 folds for cross-validation.
data = Dataset.load_from_df(baseline_df,reader)

# Baselineonly model
algo = BaselineOnly()
# Performance
perf_baseline = cross_validate(algo, data, measures=['RMSE', 'MAE'])
print(perf_baseline)
