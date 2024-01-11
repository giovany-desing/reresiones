import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

icfes = pd.read_csv('/Users/giovanysamaca/Desktop/icfes_git/data.csv')

icfes = icfes.sample(7000)




plt.figure(figsize=(10, 6))
sns.relplot(data=icfes, x="PUNT_GLOBAL", y="MOD_INGLES_PNAL")
plt.title('Correlacion en puntaje global e ingles')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()

pd.options.display.float_format = '{:,.0f}'.format

#icfes.to_csv('icfes.csv', index=False)

raw_data = icfes.drop(['GRUPO'],axis=1)

target = icfes['GRUPO']

preprocessed_data = StandardScaler().fit_transform(raw_data)




X_train, X_test, y_train, y_test = train_test_split(raw_data, target, test_size=0.3, random_state=0)


model = DecisionTreeClassifier()
# Aplico bagging al algoritmo de clasificacion
bag_class = BaggingClassifier(base_estimator=model, n_estimators=50).fit(X_train, y_train)

# Predicciones

predictions = bag_class.predict(X_test)

# Resultado
print('SCORE Bagging with Support Vector Classification {}'.format(accuracy_score(predictions, y_test)))

