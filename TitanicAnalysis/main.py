import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
##  https://www.youtube.com/watch?v=VCJdg7YBbAQ&list=PL9ooVrP1hQOEPjeOixXeBo1qFaVQRdivC
titanic_data = pd.read_csv("resources/Titanic.csv")
print("Actual size of data -> ", titanic_data.shape)
## Data Analysis
print(titanic_data.head())
sns.countplot(x="Survived", data=titanic_data)
plt.show()
sns.countplot(x="Survived", hue="Embarked", data=titanic_data)
plt.show()
# sns.joiningntplot(x="Survived", hue="Pclass", data=titanic_data)
titanic_data["Age"].hist(bins=20, figsize=(10, 5))
sns.countplot(x="SibSp",  data=titanic_data)
titanic_data["SibSp"].unique()
plt.pie()

## Data wrangling Data cleaning
titanic_data.isnull()
titanic_data.isnull().sum()

sns.heatmap(titanic_data.isnull(), yticklabels=False, xticklabels=True, cmap="Wistia")
sns.boxplot(x="Pclass", y="Age", data=titanic_data)
## Remove null values and  confirm
titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()
# for categorical regression create discrite
dumySex = pd.get_dummies(titanic_data["Sex"])