

"""### The Intriguing World of Social Media Surveillance

Social media surveillance is a hot topic in today's digital age. With the vast amount of data generated every second, understanding the trends, patterns, and implications of this surveillance is crucial. This dataset from Kaggle provides a fascinating glimpse into the academic research surrounding social media surveillance. Let's dive in and see what insights we can uncover. If you find this notebook useful, please upvote it.
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset
file_path = '960scopus.csv'
df = pd.read_csv(file_path)
df.head()

"""### Data Overview

Let's take a closer look at the structure of our dataset and understand the types of data we are dealing with.
"""

# Displaying basic information about the dataset
#df.info()

"""### Data Cleaning and Preprocessing

Before diving into the analysis, we need to clean and preprocess the data. This includes handling missing values, converting data types, and extracting relevant features.
"""

# Handling missing values
df = df.dropna(subset=['Title', 'Year', 'Cited by', 'Link', 'Abstract', 'Language of Original Document', 'Document Type', 'Publication Stage', 'Source', 'EID'])

# Converting data types
df['Year'] = df['Year'].astype(int)
df['Cited by'] = df['Cited by'].astype(int)

# Displaying the cleaned dataset
df.head()

"""### Exploratory Data Analysis (EDA)

Let's explore the dataset to uncover interesting patterns and trends. We'll start with some basic visualizations.
"""

# Distribution of publications over the years
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=df, palette='viridis')
plt.title('Number of Publications Over the Years')
plt.xticks(rotation=45)
plt.show()

# Top 10 most cited papers
top_cited = df.nlargest(10, 'Cited by')
plt.figure(figsize=(10, 6))
sns.barplot(x='Cited by', y='Title', data=top_cited, palette='viridis')
plt.title('Top 10 Most Cited Papers')
plt.show()

"""### Correlation Analysis

Let's examine the correlation between numeric variables in the dataset.
"""

# Selecting numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

"""### Predictive Analysis

Given the data, it might be interesting to predict the number of citations a paper will receive based on its features. Let's build a simple linear regression model to predict the 'Cited by' count.
"""

from sklearn.linear_model import LogisticRegression
import pickle
# Selecting features and target variable
X = df[['Year']]
y = df['Cited by']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

 # Save the model
with open('model.pkl', 'wb') as f:
  pickle.dump(model, f)


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse, r2

from flask import Flask, request, jsonify
import numpy as np
import pickle  # or pickle if you're using that
import psutil
import os

app = Flask(__name__)

# Load your trained model (adjust filename as needed)
with open('model.pkl', 'rb') as f:
  model = pickle.load(f)

@app.route('/')
def home():
    return "✅ Flask ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        prediction = model.predict([data])
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/health', methods=['GET'])
def health():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    cpu_usage = process.cpu_percent(interval=0.5)

    return jsonify({
        'memory_usage_mb': round(memory_info.rss / 1024 / 1024, 2),
        'cpu_usage_percent': cpu_usage
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






"""### Future Analysis

There are numerous directions we could take for further analysis. For instance, we could explore the impact of different document types on citation counts, or analyze the effect of open access on the number of citations. What do you think would be useful to explore next?

### Conclusion

In this notebook, we explored the fascinating dataset on social media surveillance, performed some exploratory data analysis, and built a simple predictive model. The insights gained here are just the tip of the iceberg. If you found this notebook useful, please upvote it.

## Credits
This notebook was created with the help of [Devra AI data science assistant](https://devra.ai/ref/kaggle)
"""
