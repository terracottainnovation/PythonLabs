import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# https://www.kaggle.com/shivamb/netflix-shows?select=netflix_titles.csv
data = pd.read_csv("netflix_titles.csv")
dataframe = pd.DataFrame(data)
print(dataframe.head(50))

# (dataframe.mesh.str.split(',', expand=True)
#    .stack()
#    .str.get_dummies()
#    .sum(level=0))