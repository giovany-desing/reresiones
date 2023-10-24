import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
    
from sklearn.linear_model import LogisticRegression 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

icfes = pd.read_csv('/Users/giovanysamaca/Desktop/git proyecto icfes/data.csv')

icfes = icfes.sample(3000)

pd.options.display.float_format = '{:,.0f}'.format

#icfes.to_csv('icfes.csv', index=False)

raw_data = icfes.drop(['GRUPO'],axis=1)

target = icfes['GRUPO']

preprocessed_data = StandardScaler().fit_transform(raw_data)




X_train, X_test, y_train, y_test = train_test_split(raw_data, target, test_size=0.3, random_state=0)

estimators = {
       'LogisticRegression' : LogisticRegression(),
        'SVC' : SVC(),
        'LinearSVC' : LinearSVC(),
        'SGD' : SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        'KNN' : KNeighborsClassifier(),
        'DecisionTreeClf' : DecisionTreeClassifier(),
        'RandomTreeForest' : RandomForestClassifier(random_state=0),
        'Boosting' : GradientBoostingClassifier(n_estimators=50)
        
    }


for name, estimator in estimators.items():
    bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train, y_train)
    bag_predict = bag_class.predict(X_test)
    print('='*64)
    print('SCORE Bagging with {} : {}'.format(name, accuracy_score(bag_predict, y_test)))


