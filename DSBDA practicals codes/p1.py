import pandas as pd
import numpy as np

df = pd.read_csv("autodata.csv")
print(df.head())
print("-------------------------------------------")
print(df.info())
print("-------------------------------------------")
print(df.describe())
print("-------------------------------------------")
print(df["stroke"].describe())
print("-------------------------------------------")
# print(df.isnull())
# print("-------------------------------------------")
print(df.isnull().sum())
print("-------------------------------------------")
print(df.notnull().sum())
print("-------------------------------------------")
null_rows_all = df.isnull().any(axis=1)  # for all columns
print(df[null_rows_all].index)
print("-------------------------------------------")
print(df["horsepower"].iloc[126:131])
print("-------------------------------------------")
null_rows = df["horsepower"].isnull()  # only for column horsepower
print(df[null_rows].index)
print("-------------------------------------------")
avg_hp = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", avg_hp)
df["horsepower"].replace(np.nan, avg_hp, inplace=True)
# print(df["horsepower"])
print("-------------------------------------------")
avg_rpm = df["peak-rpm"].astype("float").mean(axis=0)
print("Average of peak-rpm:", avg_rpm)
df["peak-rpm"].replace(np.nan, avg_hp, inplace=True)
print("-------------------------------------------")
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke: ", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)
print("-------------------------------------------")
df.dropna(subset=["horsepower-binned"], axis=0, inplace=True)
print("Horsepower-binned columns null value is dropped")
df.reset_index(drop=True, inplace=True)
print("-------------------------------------------")
null_rows_all = df.isnull().any(axis=1)  # for all columns
print(df[null_rows_all].index)
print("-------------------------------------------")
print(df.isnull().sum())
print("-------------------------------------------")
print(df["aspiration"].value_counts())
print("-------------------------------------------")
dummy_variable_1 = pd.get_dummies(df["aspiration"])
print(dummy_variable_1.head(10))
print("-------------------------------------------")
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("aspiration", axis=1, inplace=True)
print(df.head())
print("-------------------------------------------")
