# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV #grid search en iyi modeli bulmak için kullanılır.
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

import warnings #hata mesajlarını gizlemek için kullanılır.
warnings.filterwarnings("ignore")

# import data and EDA --> verinin yüklenmesi ve keşifsel veri analizi...

#loading data
df = pd.read_csv("diabetes.csv") 
df_name = df.columns
#df.info()
# describe = df.describe()
# print(describe)

plt.figure()
sns.pairplot(df, hue="Outcome") #outcome değişkenine göre verileri çizdirir.
plt.show()

def plot_correlation_heatmap(df): #correlation feature heatmap çizdirmek için kullanılır. heatmap ile görselleştirmek için 
    corr_matrix = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix) # correlation matrix ile görselleştirmek için 
    plt.title("Correlation Feature")
    plt.show()

plot_correlation_heatmap(df)







# outlier detection (aykırı değerler)
# train - test split
# standardisyon
# model training and evaluation (değerlendirme)
# hyperparameter tuning
# model testing with real data
