import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("3) sentiment dataset.csv")

# CORRELATION ANALYSIS

# with time features
corr_with_time = df.select_dtypes(include=['int64', 'float64'])

corr_matrix_with_time = corr_with_time.corr()

print("\n Correlation With Time Features\n")
print(corr_matrix_with_time)

#   Without time features
# Drop time columns
df_no_time = df.drop(columns=['Year', 'Month', 'Day', 'Hour'], errors='ignore')

corr_matrix = df_no_time.select_dtypes(include=['int64', 'float64']).corr()

print("\n Correlation Without Time Features\n")
print(corr_matrix)

print("\n Exploratory Data Analysis Completed")

