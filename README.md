# review-sentiment-analysis
The reviews, star-rating are the accessories of a product which describes the customer's sentiments. In this assignment TF-IDF text representation and several classification techniques are used to analyze customer sentiments.

# Libraries used:
pandas - data manipulation
nltk - natural language processing tasks
re - regular expressions
bs4 - Beautiful Soup to handle HTML tags
contractions - to fix word contractions
sklearn - for classification, regression and svm algorithms

# Data Preparation
Dataset:
https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv
.gz

The dataset contains review data of a Kitchen appliance with Ratings and Customer Reviews. The data is read into a pandas data frame for processing, The required fields for analysis are “star_rating” and “review_body”. To label the reviews, we are using the information in the rating. The ratings 4 and 5 are labelled as positive sentiment - label 1, and ratings 1 and 2 are considered negative sentiment - label 0. The neutral rating 3 is discarded from the analysis.

# Data Cleaning:
In order to extract features from the reviews it is important to clean the data and get rid of
unnecessary and redundant information. The following steps are being performed -
1. Conversion to lowercase - using pandas str.lower() function
2. Removing HTML Tags and URLs - using bs4 and regular expressions to remove
http/https strings
3. Strip Whitespaces - using str.strip()
4. Fixing Contractions - using contractions library in python
5. Removing non-alphabetical characters - using regular expressions
6. Removing extra spaces between the words

Average Length of Reviews before cleaning = 323.071085
Average Length of Reviews after cleaning = 311.87936

# Data Preprocessing
Performing lemmatization which is grouping together the inflected forms of a word so they
can be analysed as a single item and removal of stopwords such as ‘the’, ‘for’, ‘are’ which do
not account much to the emotion in the review.
1. Removal of stopwords - using nltk corpus of stopwords
2. Lemmatization - using nltk WordNetLemmatizer

Average Length of Reviews before preprocessing = 311.87936
Average Length of Reviews after preprocessing = 189.93782

# TF-IDF Feature Extraction
TF-IDF Term Frequency and Inverse Document Frequency is intended to reflect how
important a word is to a document in a collection or corpus.
Using TfidfVectorizer in the sklearn feature extraction library TD-IDF vectors for both test and
train split are being generated.
Train:Test Split = 0.2 and is done using sklearn.
Once the feature extraction is done the following models are trained and tested.
The metrics obtain on the test and training split after fitting the curve are as follows -
  # 1. Perceptron
  Training Split Metrics -
  Accuracy = 0.902575
  Label ‘1’ Metrics -
  Precision = 0.9163555084691016
  Recall = 0.8861305113757043
  F1 Score = 0.9009895959044195

  Test Split Metrics -
  Accuracy = 0.8567
  Label ‘1’ Metrics -
  Precision = 0.8714949610986371
  Recall = 0.8361304543860528
  F1 Score = 0.8534465125792596

  # 2. SVM
  Training Split Metrics -
  Accuracy = 0.93375
  Label ‘1’ Metrics -
  Precision = 0.9344553588187449
  Recall = 0.9330076587663514
  F1 Score = 0.9337309476474486

  Test Split Metrics -
  Accuracy = 0.896075
  Label ‘1’ Metrics -
  Precision = 0.9004662477194405
  Recall = 0.8901357647412454
  F1 Score = 0.8952712065099638

  # 3. Logistic Regression
  Training Split Metrics -
  Accuracy = 0.91381875
  Label ‘1’ Metrics -
  Precision = 0.916782002566748
  Recall = 0.910356201351841
  F1 Score = 0.9135578026166491

  Test Split Metrics -
  Accuracy = 0.899225
  Precision = 0.9054673182651192
  Recall = 0.8910876208606783
  F1 Score = 0.8982199217270546

  # 4. Naive Bayes (Multinomial Bayes)
  Training Split Metrics -
  Accuracy = 0.8879375
  Label ‘1’ Metrics -
  Precision = 0.8897451022226684
  Recall = 0.8857432001899074
  F1 Score = 0.8877396411174695
  Test Split Metrics -
  Accuracy = 0.87015
  Precision = 0.8747398873268031
  Recall = 0.863433695706628
  F1 Score = 0.8690500201694231

SVM and Logistic Regression give better predictions on the data compared to Naive Bayes and Perceptron. Accuracy may be improved with better data cleaning and preprocessing methods and tuning.
