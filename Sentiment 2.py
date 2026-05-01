import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("3) sentiment dataset.csv")

# EXPLORATORY DATA ANALYSIS (EDA)
print("\n Statistic Summary\n")
print(df.describe())

print("\nSentiment Distribution:\n")
print(df['Sentiment'].value_counts())

# Engagement by Sentiment

print("\nAverage Engagement by Sentiment:\n")
print(df.groupby('Sentiment')[['Likes','Retweets']].mean())

# VISUALIZATIONS

# Sentiment Count
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()

# Likes Distribution
df['Likes'].hist()
plt.title('Likes Distribution')
plt.show()

# Boxplot (Likes vs Sentiment)
sns.boxplot(x='Sentiment', y='Likes', data=df)
plt.title('Likes by Sentiment')
plt.show()

# Retweets vs Likes Scatter
plt.scatter(df['Likes'], df['Retweets'])
plt.xlabel("Likes")
plt.ylabel("Retweets")
plt.title("Likes vs Retweets")
plt.show()
