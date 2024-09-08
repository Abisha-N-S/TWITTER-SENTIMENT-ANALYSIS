from zipfile import ZipFile
dataset="D:\\Data Science\\Project7\\sentiment140.zip"

with ZipFile(dataset, 'r') as zip:
   zip.extractall()
   print('The data is extracted ')

#importing all the packages

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#printing the stopwords
print(stopwords.words('english'))

#Datapreprocessing

#loading the data from CSV file to pandas dataframe
twitter_data=pd.read_csv("D:\\Data Science\\Project7\\training.1600000.processed.noemoticon.csv",encoding='ISO-8859-1')

twitter_data.head()

#naming the column and reading the dataset again
column_names=['target','id','date','flag','user','text']
twitter_data=pd.read_csv("D:\\Data Science\\Project7\\training.1600000.processed.noemoticon.csv",names=column_names,encoding='ISO-8859-1')

twitter_data.head()

# counting the number of missing values
twitter_data.isnull().sum()

#converting the target column "4" to "1"

twitter_data.replace({'target':{4:1}},inplace=True)

#checking the distribution of the target column
twitter_data['target'].value_counts()

port_stem=PorterStemmer()

def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content=' '.join(stemmed_content)

  return stemmed_content

twitter_data['stemmed_content']=twitter_data['text'].apply(stemming)

twitter_data.head()

print(twitter_data['stemmed_content'])

print(twitter_data['target'])

# seperating the data and label
x=twitter_data['stemmed_content'].values
y=twitter_data['target'].values

print(x)

print(y)

#splitting the dataset to training and testing

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,stratify=y,random_state=2)

print(x.shape,x_train.shape,x_test.shape)

print(x_train)

print(x_test)

#converting the text data into numeric
vectorizer=TfidfVectorizer()
x_train=vectorizer.fit_transform(x_train)
x_test=vectorizer.transform(x_test)

print(x_train)

print(x_test)

#USING THE MACHINE LEARNING MODEL
#LOGISTIC REGRESSION


model=LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)

#MODEL EVALUATION

#accuracy score on the training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_prediction)

print('Accuracy score on the training data:',training_data_accuracy)

#accuracy score for the testing data
x_test_prediction=model.predict(x_test)
testing_data_accuracy=accuracy_score(y_test,x_test_prediction)

print('Accuracy score for the testing data:',testing_data_accuracy)

#MODEL ACCURACY

import pickle

name="trained_model.pkl"
pickle.dump(model,open(name,'wb'))

"""##using the saved model for future predictions"""

#loading the saved model

loaded_model=pickle.load(open("/content/trained_model.pkl",'rb'))

x_new=x_test[100]
print(y_test[100])

prediction=loaded_model.predict(x_new)
print(prediction)

if (prediction[0]==0):
  print("negative tweet")

else:
  print("positive tweet")