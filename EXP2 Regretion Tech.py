

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("train.csv")
dataset.head()

dataset.describe()

dataset.info()

dataset.isnull().sum()

# dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
# df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

#PClass can be a proxy for socio-economic status (SES)
sb.boxplot(x="Pclass",y="Age",data=dataset,palette=sb.color_palette('bright')[:3])

sb.displot(data=dataset, x='Age', hue='Survived', kind='kde', fill=True, palette=sb.color_palette('bright')[:3], height=5, aspect=1.5)
plt.title('Age Distribution by Survived')
plt.show()

"""sb.displot(data=dataset, x='Age', hue='Pclass', kind='kde', fill=True, palette=sb.color_palette('bright')[:3], height=5, aspect=1.5)
plt.title('Age Distribution by Pclass')
plt.show()
"""

sb.displot(data=dataset, x='Age', hue='Pclass', kind='kde', fill=True, palette=sb.color_palette('bright')[:3], height=5, aspect=1.5)
plt.title('Age Distribution by Pclass')
plt.show()

"""df_corr = dataset.drop(['PassengerId'], axis = 1)
sb.heatmap(df_corr.corr(),annot = True, cmap = plt.cm.Blues)
plt.title('Correlation Matrix')
"""

df_corr = dataset.drop(['PassengerId'], axis = 1)
sb.heatmap(df_corr.corr(),annot = True, cmap = plt.cm.Blues)
plt.title('Correlation Matrix')

train_data = dataset.copy()
train_data["Age"].fillna(dataset["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(dataset['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)

train_data.isnull().sum()

sb.displot(data=train_data, x='Age', hue='Survived', kind='kde', fill=True, palette=sb.color_palette('bright')[:3], height=5, aspect=1.5)
plt.title('Age Distribution by Survived')
plt.show()

sb.displot(data=train_data, x='Age', hue='Pclass', kind='kde', fill=True, palette=sb.color_palette('bright')[:3], height=5, aspect=1.5)
plt.title('Age Distribution by Pclass')
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score

x = pd.DataFrame(np.c_[train_data['Pclass'],train_data['Fare']],columns=['Age','Fare'])
y = train_data['Survived']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y =le.fit_transform(y)

x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=0,test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score

lr = LogisticRegression()

# lr.fit(x_train,y_train)
# y_predRF=lr.predict(x_test)

lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Logistic Regression",accuracy_score(y_test,y_pred))