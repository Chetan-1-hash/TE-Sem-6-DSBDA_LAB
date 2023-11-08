import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(sns.load_dataset("titanic"))
print(df.head())
print('*'*65)
print(df.describe())
print('*'*65)
print(df.info())
print('*'*65)
print(df.isnull().sum())
print('*'*65)
df['age'] = df['age'].fillna(np.mean(df['age']))
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['deck'] = df['deck'].fillna(df['deck'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
print('*'*65)
print(df.isnull().sum())
print('*'*65)
sns.displot(df['survived'])
plt.show()
sns.displot(df['age'], kde=True)
plt.show()
sns.catplot(x='pclass', y='fare', data=df, kind='bar')
plt.show()
