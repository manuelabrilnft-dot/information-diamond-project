from IPython.display import display
import pandas as pd
import pandasql as ps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('dataset.csv')

#List all numeric columns
numeric_cols = dataset.select_dtypes(include=[np.number]).columns

#Get rid of the outliers
for col in numeric_cols:
    Q1 = dataset[col].quantile(0.21)
    Q3 = dataset[col].quantile(0.79)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

#Some statistics on the dataset
print("Shape:", dataset.shape)
print("\nData Types:\n", dataset.dtypes)
print("\nMissing Values:\n", dataset.isnull().sum())
print("\nSummary Statistics:\n", dataset.describe(include='all'))

#Encode the database 
#Make all columns numerical
dataset_encoded = pd.get_dummies(dataset, drop_first=2)

dataset_encoded = dataset_encoded.astype(int)
dataset_encoded.dtypes.head(49)
display(dataset_encoded)

#List the names of all columns
print("\nColumn names:")
for col in dataset_encoded.columns:
    print(" -", col)


#Graph of the values for all features
'''
for col in numeric_cols:
    plt.figure()
    sns.boxplot(data=dataset_encoded, x=col)
    plt.title(f"Outlier Check: {col}")
    plt.show()
'''

#Exploratory queries

def query(sql):
    return ps.sqldf(sql, globals())

print(query("""
SELECT job, AVG(credit_amount) AS avg_credit
FROM dataset
GROUP BY job
ORDER BY avg_credit DESC;
"""))

print("")

print(query("""
SELECT housing, AVG(credit_amount) AS avg_credit
FROM dataset
GROUP BY housing
ORDER BY avg_credit DESC;
"""))

print("")

print(query("""
SELECT personal_status, AVG(credit_amount) AS avg_credit
FROM dataset
GROUP BY personal_status
ORDER BY avg_credit DESC;
"""))

dataset_encoded.to_csv("query_result.csv", index=False)