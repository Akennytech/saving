# social_media_surveillance_analysis_and_predictions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
file_path = '960scopus.csv'
df = pd.read_csv(file_path)

# Clean data
df = df.dropna(subset=['Title', 'Year', 'Cited by', 'Link', 'Abstract',
                       'Language of Original Document', 'Document Type',
                       'Publication Stage', 'Source', 'EID'])

df['Year'] = df['Year'].astype(int)
df['Cited by'] = df['Cited by'].astype(int)

# Plot: Publications over years
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=df, palette='viridis')
plt.title('Number of Publications Over the Years')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('publications_over_years.png')
plt.close()

# Plot: Top 10 cited papers
top_cited = df.nlargest(10, 'Cited by')
plt.figure(figsize=(10, 6))
sns.barplot(x='Cited by', y='Title', data=top_cited, palette='viridis')
plt.title('Top 10 Most Cited Papers')
plt.tight_layout()
plt.savefig('top_cited_papers.png')
plt.close()

# Plot: Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Predictive model
X = df[['Year']]
y = df['Cited by']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
