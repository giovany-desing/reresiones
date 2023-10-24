import pandas as pd

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor, Lasso, Ridge
)
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics  import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")

icfes = pd.read_csv('/Users/giovanysamaca/Desktop/git proyecto icfes/data.csv')

icfes = icfes.sample(3000)

#plt.figure(figsize=(10, 6))
#sns.histplot(data=icfes, x='PUNT_GLOBAL', hue='GRUPO', multiple='stack')
#plt.title('Clustering con MiniBatchKMeans')
#plt.xlabel('Feature1')
#plt.ylabel('Feature2')
#plt.show()

X = icfes.drop(['PUNT_GLOBAL'], axis=1)
y = icfes[['PUNT_GLOBAL']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

estimadores = {
        'SVR' : SVR(gamma = 'auto', C=1.0, epsilon=0.1),
        'RANSAC' : RANSACRegressor(),
        'HUBER' : HuberRegressor(epsilon=1.35),
        'LASSO' : Lasso(alpha=0.02),
        'RIDGE' : Ridge(alpha=1)
    }

for name, estimator in estimadores.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        print("="*64)
        print(name)
        print("MSE: ", mean_squared_error(y_test, predictions))
        print("Score", estimator.score(X_test, y_test))