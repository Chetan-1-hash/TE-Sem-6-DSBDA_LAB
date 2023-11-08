import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("Social_Network_Ads.csv")
print(df.head())
print("\n------------------------------------------------------------------------------")
print(df.describe())
print("\n------------------------------------------------------------------------------")
print(df.isnull().sum())
print("\n------------------------------------------------------------------------------")
sns.countplot(data=df, x='Purchased')
plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
print("\n------------------------------------------------------------------------------")
features = df[['Age', 'EstimatedSalary']]
label = df['Purchased']

scaler = StandardScaler()
features = scaler.fit_transform(features)

x = features
y = label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
model = LogisticRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
print("\n------------------------------------------------------------------------------")
y_pred = model.predict(x_test)
print(y_pred)
print("\n------------------------------------------------------------------------------")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
print("\n------------------------------------------------------------------------------")
def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0][0], cm[0][1], cm[1][0], cm[1][1]


TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)
print("\n------------------------------------------------------------------------------")
print("The Accuracy is ", (TP + TN) / (TP + TN + FP + FN))
print("The precision is ", TP / (TP + FP))
print("The recall is ", TP / (TP + FN))
