import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import category_encoders as cat_encoder
import seaborn as sns
import plotly.express as px
from langdetect import detect
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import re
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
# ----------------------------------------------   import datasets   ---------------------------------------------- #

csv_table=pd.read_table('mediaeval-2015-trainingset.txt',sep='\t')
csv_table.to_csv("mediaeval-2015-trainingset.csv",index=False)
training_data= pd.read_csv("mediaeval-2015-trainingset.csv")

csv_table2=pd.read_table('mediaeval-2015-testset.txt',sep='\t')
csv_table2.to_csv("mediaeval-2015-testset.csv",index=False)
testing_data= pd.read_csv("mediaeval-2015-testset.csv")

test= pd.read_csv("test.csv")

# --------------------------------------------   clean training data   -------------------------------------------- #

# remove NA values 
training_data.dropna(axis='index',how='any')

# convert ground truth labels
ground_truth_codes = {'fake':0,'humor':0, 'real':1}
training_data['label'] = training_data['label'].map(ground_truth_codes)

# remove imageId
training_data = training_data.drop('imageId(s)',axis=1) 

# oversampling to fix class imbalance
overSampler = RandomOverSampler(random_state=0)
new_training_data, new_training_labels = overSampler.fit_resample(training_data, training_data['label'])

# function to clean tweetText data itself
def clean_tweetText(textToClean):

    #removing symbols/punction/emoticons
    textToClean = re.sub("[^a-zA-Z ]+", "", textToClean) 
    #make tweetText data lowercase
    textToClean = textToClean.lower() 
    #removing URLs
    textToClean = re.split('http.*', str(textToClean))[0] 

    return textToClean


# function to find language of each tweetText 
def find_language_count(textToCheck):

    try:
        # try detect the language
        lang = detect(textToCheck)
    except:
        # if not detected, label as 'other'
        lang = 'other'

    return lang


#function to remove stopwords in each tweetText
stopword_list = stopwords.words('english')

def remove_stopwords(textToCheck):
    return ' '.join([wordToCheck for wordToCheck in textToCheck.split() if wordToCheck not in (stopword_list)])


# apply text cleaning functions
new_training_data['language'] = new_training_data['tweetText'].apply(find_language_count)
new_training_data['tweetText'] = new_training_data['tweetText'].apply(clean_tweetText)
new_training_data['tweetText'] = new_training_data['tweetText'].apply(remove_stopwords)

# print(new_training_data.head(10))

# -------------------------------------------  data visualisation  -------------------------------------------- #

# plot number of posts in each langauge -----------> change this to pie chart
# lang_plot = sns.countplot(x="language", hue="label", data=new_training_data,palette='Set2')
# plt.show()

# plot number of posts in each class for original dataset
# training_data.groupby(['label'])['tweetText'].count().plot(kind='bar')
# label_plot = sns.countplot(x="label", data=training_data,palette='Set2')
# plt.show()

# plot number of posts in each class for oversampled dataset
# new_training_data.groupby(['label'])['tweetText'].count().plot(kind='bar')
# label_plot = sns.countplot(x="label", data=new_training_data,palette='Set2')
# plt.show()


# plot tweet lengths


# plot common words



# --------------------------------------------   clean test data   -------------------------------------------- #
# test data will not be oversampled

# remove NA values
testing_data.dropna(axis='index',how='any')

# convert ground truth labels
ground_truth_codes = {'fake':0,'humor':0, 'real':1}
testing_data['label'] = testing_data['label'].map(ground_truth_codes)

# remove imageId
testing_data = testing_data.drop('imageId(s)',axis=1) 

# apply text cleaning functions
testing_data['tweetText'] = testing_data['tweetText'].apply(clean_tweetText)
testing_data['tweetText'] = testing_data['tweetText'].apply(remove_stopwords)

# ----------------------------------------------  classifier  ------------------------------------------------- #
training_text = new_training_data['tweetText']
training_label = new_training_data['label']
testing_text = testing_data['tweetText']
testing_label = testing_data['label']

# apply tokenising
count_vectoriser = CountVectorizer(stop_words='english')
count_training_text = count_vectoriser.fit_transform(training_text)
count_testing_text = count_vectoriser.transform(testing_text)

# train classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(count_training_text, training_label)
test_data_prediction = naive_bayes.predict(count_testing_text)

# --------------------------------------------  result analysis  ----------------------------------------------- #

# Recall
r_score = metrics.recall_score(testing_label, test_data_prediction)
print("Recall:   %0.3f" % r_score)

# Precision
p_score = metrics.precision_score(testing_label, test_data_prediction)
print("Precision:   %0.3f" % p_score)

# F1 Score
f_score = metrics.f1_score(testing_label, test_data_prediction)
print("F1 score:   %0.3f" % f_score)

# Accuracy
a_score = metrics.accuracy_score(testing_label, test_data_prediction)
print("Accuracy:   %0.3f" % a_score)

# Confusion Matrix
# conf_matrix = confusion_matrix(testing_data, test_data_prediction)

# print(cm5)
# sns.heatmap(my_df.corr())
# print(conf_matrix)

