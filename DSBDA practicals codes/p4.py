import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("boston.csv")
print(data.head())
print("\n----------------------------------------------------------------------")
print(data.tail())
print("\n----------------------------------------------------------------------")
print(data.shape)
print("\n----------------------------------------------------------------------")
print(data.isnull().sum())
print("\n----------------------------------------------------------------------")
x = data.iloc[:, 0:13]
y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=4)
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)
print(reg.score(X_test, Y_test))
y_pred = reg.predict(X_test)
sns.regplot(x = Y_test, y = y_pred, ci= 95)
plt.show()
