import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# print(sns.get_dataset_names())
titanic = sns.load_dataset("titanic")
data = pd.DataFrame(titanic)
# print(titanic.head()) # for seaborn
print(data.head())
print("\n----------------------------------------------------------------")
print(data.info())
print("\n----------------------------------------------------------------")
print(data.describe())
print("\n----------------------------------------------------------------")
print(data.isnull().sum())
print("\n----------------------------------------------------------------")
data['age'] = data['age'].fillna(np.mean(data['age']))
data['deck'] = data['deck'].fillna(data['deck'].mode()[0])
data['embark_town'] = data['embark_town'].fillna(data['embark_town'].mode()[0])
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
print("\n----------------------------------------------------------------")
print(data.isnull().sum())
sns.boxplot(x=data['sex'], y=data["age"], hue=data["survived"], palette='Set2').set_title('Plot for distribution of age with respect to each gender along with the information about whether they survived or not')
plt.show()
