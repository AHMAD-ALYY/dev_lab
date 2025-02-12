import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv("netflix_titles.csv")

# Convert 'date_added' to datetime format
df['date_added'] = pd.to_datetime(df['date_added'], format='mixed', errors='coerce') 

# Fill missing values
df['director'].fillna('Unknown', inplace=True)
df['cast'].fillna('Unknown', inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['rating'].fillna('Not Rated', inplace=True)
df['duration'].fillna('Unknown', inplace=True)

df['date_added'] = pd.to_datetime(df['date_added'], dayfirst=True, errors='coerce')

# Summary statistics
print("Dataset Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe(include='all'))

# --- Visualizations ---
plt.style.use('ggplot')

# 1. Count of Movies vs TV Shows
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=df, palette='coolwarm')
plt.title("Count of Movies vs TV Shows")
plt.show()

# 2. Top 10 Countries with Most Titles
plt.figure(figsize=(10, 5))
df['country'].value_counts().head(10).plot(kind='bar', color='royalblue')
plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 3. Distribution of Release Years
plt.figure(figsize=(12, 5))
sns.histplot(df['release_year'], bins=30, kde=True, color='purple')
plt.title("Distribution of Release Years")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()

# 4. Trend of Titles Added Over Time
plt.figure(figsize=(12, 5))
df['date_added'].dt.year.value_counts().sort_index().plot(kind='line', marker='o', color='darkred')
plt.title("Trend of Titles Added Over Time")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# 5. Genre Distribution (Listed in)
plt.figure(figsize=(12, 6))
genres = df['listed_in'].str.split(', ').explode()
genres.value_counts().head(15).plot(kind='bar', color='teal')
plt.title("Top 15 Most Common Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 6. Word Cloud for Movie Titles
plt.figure(figsize=(10, 5))
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(" ".join(df['title']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Netflix Titles")
plt.show()


print("EDA and visualizations completed successfully!")
