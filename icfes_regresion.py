import pandas as pd

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics  import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")

icfes = pd.read_csv('/Users/giovanysamaca/Desktop/icfes_git/data.csv')

icfes = icfes.sample(6000)


X = icfes.drop(['PUNT_GLOBAL'], axis=1)
y = icfes[['PUNT_GLOBAL']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

model = Ridge(alpha=0.02)
model.fit(X_train,y_train)

print("Score Ridge", model.score(X_test, y_test))


#grafico la cantidad de estudiantes por genero
fig, ax = plt.subplots()

gener = icfes['ESTU_GENERO'].unique()
counts = icfes['ESTU_GENERO'].value_counts()
bar_labels = ['Hombre', 'Mujer']
bar_colors = ['tab:red', 'tab:blue']

ax.bar(gener, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('Estudiantes')
ax.set_title('Genero de estudiantes')
ax.legend(title='Genero')

plt.show()
