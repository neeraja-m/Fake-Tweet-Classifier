import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import category_encoders as cat_encoder
import seaborn as sns
import plotly.express as px
from langdetect import detect
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import re
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
ground_truth_codes = {'fake':'fake','humor':'fake', 'real':'real'}
training_data['label'] = training_data['label'].map(ground_truth_codes)

# remove imageId
training_data = training_data.drop('imageId(s)',axis=1) 

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

training_data['language'] = training_data['tweetText'].apply(find_language_count)
training_data['tweetText'] = training_data['tweetText'].apply(clean_tweetText)

print(training_data.head())

#removing stopwords
# stopword_list = stopwords.words('english')
# training_data['tweetText'] = training_data['tweetText'].apply(lambda x: ' '.join([wordToCheck for wordToCheck in x.split() if wordToCheck not in (stopword_list)]))

# -------------------------------------------  data visuaisation   -------------------------------------------- #

# plot number of posts in each langauge -----------> change this to pie chart
# lang_plot = sns.countplot(x="language", hue="label", data=training_data,palette='Set2')
# plt.show()

# plot number of posts in each class
# training_data.groupby(['label'])['tweetText'].count().plot(kind='bar')
# label_plot = sns.countplot(x="label", data=training_data,palette='Set2')
# plt.show()

# plot tweet lenghts?


# plot 



# --------------------------------------------   clean test data   -------------------------------------------- #
# testing_data['tweetText'] = testing_data['tweetText'].apply(clean_tweetText)

# ----------------------------------------------  classifier  ------------------------------------------------- #

# cv = CountVectorizer(max_features = 5000)
# X = cv.fit_transform(corpus).toarray()
# y = training_data.iloc[0:40000, 2].values

# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier.fit(training_data, testing_data)

# # Predicting the Test set results
# test_prediction = classifier.predict(testing_data)

# --------------------------------------------  result analysis  ----------------------------------------------- #

# Precision



# Recall



# F1 Score



# Confusion Matrix
# confusion_matrix = confusion_matrix(testing_data, test_prediction)
# my_df = pd.DataFrame(training_data)
# sns.heatmap(my_df.corr())
# # print(confusion_matrix)

