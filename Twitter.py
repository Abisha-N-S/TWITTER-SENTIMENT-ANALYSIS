import os
import shutil
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from zipfile import ZipFile

# Define the path to the kaggle.json file
kaggle_json_path ="C:\\Users\\abish\\Downloads\\kaggle.json"

# Create the .kaggle directory if it doesn't exist
kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Copy the kaggle.json file to the .kaggle directory
shutil.copy(kaggle_json_path, kaggle_dir)

# Set file permissions (not necessary on Windows, but can be kept)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# Unzipping the dataset
dataset = "D:\\Data Science\\Project7\\sentiment140.zip"
with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('The data is extracted')

# Download NLTK stopwords if not available
nltk.download('stopwords')

# Print stopwords
print(stopwords.words('english'))

# Load dataset
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv("D:\\Data Science\\Project7\\training.1600000.processed.noemoticon.csv", names=column_names, encoding='ISO-8859-1')

twitter_data.replace({'target': {4: 1}}, inplace=True)
# Porter Stemmer
port_stem = PorterStemmer()

def stemming(content):
    # Remove special characters and non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert to lowercase and split into words
    stemmed_content = stemmed_content.lower().split()
    # Stemming
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Apply stemming function to the 'text' column
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

# Separating the data and labels
x = twitter_data['stemmed_content'].values
y = twitter_data['target'].values

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, stratify=y, random_state=2)

print(x.shape,x_train.shape,x_test.shape)

# Converting text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Model evaluation on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score on the training data:', training_data_accuracy)

# Model evaluation on testing data
x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score for the testing data:', testing_data_accuracy)

# Save the trained model
name = "trained_model.pkl"
pickle.dump(model, open(name, 'wb'))

# Load the saved model and make predictions
loaded_model = pickle.load(open("trained_model.pkl", 'rb'))

# Make a prediction for a single example
x_new = x_test[100]
print("Actual label:", y_test[100])

prediction = loaded_model.predict([x_new])
print("Prediction:", prediction)

if prediction[0] == 0:
    print("negative tweet")
else:
    print("positive tweet")