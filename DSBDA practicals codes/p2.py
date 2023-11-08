import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

data = {
    'name': pd.Series(['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack',
                       'Katie', 'Liam', 'Mia', 'Nate', 'Olivia', 'Peter', 'Quinn', 'Rachel', 'Sam', 'Tyler']),
    'division': pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C',
                           'B', 'A', 'C']),
    'marks1': pd.Series([70, 80, 85, 90, 95, 65, 75, 60, 50, 85, np.nan, 55, 80, 70, 75, 40, 90, 80, 85, 65]),
    'marks2': pd.Series([60, 70, 75, 80, 85, 55, 65, 50, 40, 75, 80, 45, 70, 60, np.nan, 30, 80, 70, 75, 55]),
    'marks3': pd.Series([5, 60, 65, 70, 75, 45, 55, 40, 30, 65, 70, 35, 60, 50, 55, 20, 70, 60, np.nan, 45])
}

df = pd.DataFrame(data)
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())

df['marks1'].fillna(df[['marks2', 'marks3']].mean(axis=1), inplace=True)
df['marks2'].fillna(df[['marks1', 'marks3']].mean(axis=1), inplace=True)
df['marks3'].fillna(df[['marks1', 'marks2']].mean(axis=1), inplace=True)

print(df.isnull().sum())

sns.boxplot(data=data, x='marks1', color='red')
plt.show()
sns.boxplot(data=data, x='marks2', color='green')
plt.show()
sns.boxplot(data=data, x='marks3', color='yellow')
plt.show()


def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1
    print(Q1,Q3,IQR)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers


outliers = detect_outliers_iqr(data['marks3'])
print("Outliers: ", outliers)

wo_outliers = df[~df['marks3'].isin(outliers)]
print(wo_outliers)

sns.boxplot(data=wo_outliers, x='marks3', color='yellow')
plt.show()

sns.histplot(df, kde=True)
plt.show()

scaler = StandardScaler()
df[['marks1', 'marks2', 'marks3']] = scaler.fit_transform(df[['marks1', 'marks2', 'marks3']])

sns.histplot(df, kde=True)
plt.show()
