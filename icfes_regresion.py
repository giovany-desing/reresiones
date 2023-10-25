import pandas as pd

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics  import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")

icfes = pd.read_csv('/Users/giovanysamaca/Desktop/icfes_git/data.csv')

icfes = icfes.sample(5000)

#plt.figure(figsize=(10, 6))
#sns.histplot(data=icfes, x='PUNT_GLOBAL', hue='GRUPO', multiple='stack')
#plt.title('Clustering con MiniBatchKMeans')
#plt.xlabel('Feature1')
#plt.ylabel('Feature2')
#plt.show()

X = icfes.drop(['PUNT_GLOBAL'], axis=1)
y = icfes[['PUNT_GLOBAL']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

model = Ridge(alpha=0.02)
model.fit(X_train,y_train)

print("Score Ridge", model.score(X_test, y_test))