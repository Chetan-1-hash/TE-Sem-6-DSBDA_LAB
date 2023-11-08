import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("iris.csv")
print(df.head())
print("\n------------------------------------------------------------------------------")
print(df.info())
print("\n------------------------------------------------------------------------------")
print(df.describe())
print("\n------------------------------------------------------------------------------")
print(df.isnull().sum())
print("\n------------------------------------------------------------------------------")
x = df.drop(['Species'], axis=1)
y = df['Species']
print(x)
print("\n------------------------------------------------------------------------------")
print(y)
print("\n------------------------------------------------------------------------------")
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print("\n------------------------------------------------------------------------------")
model = GaussianNB()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
print(model.score(X_test, Y_test))
print("\n------------------------------------------------------------------------------")
print(accuracy_score(Y_test, Y_predict))
print("\n------------------------------------------------------------------------------")
cm1 = confusion_matrix(Y_test, Y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
print("Confusion matrix: ")
print(cm1)
print("\n------------------------------------------------------------------------------")
disp.plot()
plt.show()
print("\n------------------------------------------------------------------------------")


def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0][0], cm[0][1], cm[1][0], cm[1][1]


TP, FP, FN, TN = get_confusion_matrix_values(Y_test, Y_predict)
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)
print("\n------------------------------------------------------------------------------")
print("The Accuracy is ", (TP + TN) / (TP + TN + FP + FN))
print("The precision is ", TP / (TP + FP))
print("The recall is ", TP / (TP + FN))
