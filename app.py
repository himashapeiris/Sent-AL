# import libraries
import csv
import os
import re
from datetime import timedelta
import time as time
from sqlite3.dbapi2 import Date
import numpy as np
import pandas as pd
import tweepy
from flask import Flask, request, render_template
from flask import session
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tweepy import OAuthHandler
from langdetect import detect


data = pd.read_csv(r'reviews_dataset.csv', encoding='unicode_escape')
data = data[['review', 'sentiment']]

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['review'].values)

# create OAuthHandler object
auth = OAuthHandler('4vv4n2DO6YH2iNL8nFqafD6Ku', 'NHlnw9OVdqWLEWStVlRiAN6L8BEXAWggFEgnD98pJWnlnzECtB')
# set access token and secret
auth.set_access_token('1321517957348876289-Hy2imCsnhedMibXFsf03YMs0HHNnc8',
                      'Wqgx8o84UUdaTjzlY1Gi3MlJeKCoKwZbBkWabb0qjKP47')

# create tweepy API object to fetch tweets
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# load json and create model
json_file = open('Models/SentAL_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
sental_sentiment_analyzer_model = model_from_json(loaded_model_json)
# load weights into new model
sental_sentiment_analyzer_model.load_weights("Models/SentAl_model1.h5")


# For Categorizing
# load json and create model
cat_json_file = open('Models/Categorizing_model.json', 'r')
cat_model_json = cat_json_file.read()
cat_json_file.close()
cat_model = model_from_json(cat_model_json)
# load weights into new model
cat_model.load_weights("Categorizing_model.h5")

TAG_RE = re.compile(r'<[^>]+>')

# Initialize the flask App
app = Flask(__name__)
app.secret_key = 'secret key of Sent-AL'


# preprocessing
def preprocess_text(sen):
    sentence = sen

    # removing hyperlinks
    sentence = re.sub(r"\S*https?:\S*", "", sentence)

    # remove mentions
    sentence = re.sub(r'@\S+', '', sentence, flags=re.MULTILINE)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Remove words starting with special characters
    sentence = re.sub(r'(\s)#\w+', r'\1', sentence)


    return sentence


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/', methods=['POST'])
def redirect_to_home():
    if request.method == 'POST':
        session['username'] = request.form['username']
    return render_template('home.html', user=session['username'])


@app.route('/logout', methods=['POST'])
def logout():
    if 'username' in session:
        session.pop('username', None)
        return render_template('login.html')
    else:
        return '<p>user already logged out</p>'


@app.route('/home')
def home():
    return render_template('home.html', user=session['username'])

@app.route('/twitter')
def tweets():
    return render_template('twitter.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/help')
def help():
    return render_template('help.html')


# For Twitter Data
@app.route('/twitter', methods=['POST'])
def predict_tweets():
    category1 = 0
    category2 = 0
    category3 = 0
    category4 = 0
    positive_count = 0
    negative_count = 0
    tweet_count = 0
    category1_negative_count = 0
    category1_positive_count = 0
    category2_negative_count = 0
    category2_positive_count = 0
    category3_negative_count = 0
    category3_positive_count = 0
    category4_negative_count = 0
    category4_positive_count = 0
    count = 0
    date_since = Date.today()


    if request.method == 'POST':
        search = session['username']
        days = int(request.form['days'])
        neg_tweets = []
        neg_stmnt = []
        pos_tweets = []
        pos_stmnt = []
        tweets_cat1 = []
        stmnt_cat1 = []
        tweets_cat2 = []
        stmnt_cat2 = []
        tweets_cat3 = []
        stmnt_cat3 = []
        tweets_cat4 = []
        stmnt_cat4 = []

        new_search = search + " -filter:retweets"

        date_until = Date.today()
        if days == 7:
            date_since = date_until - timedelta(days=7)
        elif days == 14:
            date_since = date_until - timedelta(days=14)
        elif days == 30:
            date_since = date_until - timedelta(days=29)
        elif days == 90:
            date_since = date_until - timedelta(days=90)

        date_since = str(date_since)
        date_until = str(date_until)
        # print("from" + date_since)
        # print("to" + date_until)

        try:

            search_results = tweepy.Cursor(api.search,
                                           q=new_search,
                                           lang="en",
                                           tweet_mode="extended",
                                           since=date_since,
                                           until=date_until,
                                           exclude_replies=True,
                                           include_entities=False).items()

            for tweet in search_results:
                count += 1
                if (tweet.user.screen_name != search.lower()) and (not tweet.retweeted) and (
                        'RT @' not in tweet.full_text):
                    print(tweet.created_at)
                    print(tweet.user.screen_name)
                    cleaned_tweet = preprocess_text(tweet.full_text)
                    preprocessed_tweet_list = []
                    preprocessed_tweet_list.clear()
                    preprocessed_tweet_list.append(cleaned_tweet)
                    tokenized_list = tokenizer.texts_to_sequences(preprocessed_tweet_list)
                    twt1 = pad_sequences(tokenized_list, maxlen=174, dtype='int32', value=0)

                    sentiment = sental_sentiment_analyzer_model.predict(twt1)
                    category = cat_model.predict(twt1)

                    if np.argmax(sentiment) == 0:
                        tweet_count = tweet_count + 1
                        negative_count = negative_count + 1
                        neg_tweets.append(tweet.full_text)
                        neg_stmnt.append('Negative')
                        # print(cleaned_tweet)
                        # print(': Neg')
                    elif np.argmax(sentiment) == 2:
                        tweet_count = tweet_count + 1
                        positive_count = positive_count + 1
                        pos_tweets.append(tweet.full_text)
                        pos_stmnt.append('Positive')
                        # print(cleaned_tweet)
                        # print(': Pos')
                    elif np.argmax(sentiment) == 1:
                        # print(cleaned_tweet)
                        print(': Neu')

                    # categories
                    if np.argmax(sentiment) == 0 or np.argmax(sentiment) == 2:
                        if np.argmax(category) == 0:
                            category1 = category1 + 1
                            tweets_cat1.append(tweet.full_text)
                            stmnt_cat1.append('4G/ Internet Homebroadband')
                            if np.argmax(sentiment) == 0:
                                category1_negative_count = category1_negative_count + 1
                            elif np.argmax(sentiment) == 2:
                                category1_positive_count = category1_positive_count + 1

                        elif np.argmax(category) == 1:
                            category2 = category2 + 1
                            tweets_cat2.append(tweet.full_text)
                            stmnt_cat2.append('Network/ Coverage')
                            if np.argmax(sentiment) == 0:
                                category2_negative_count = category2_negative_count + 1
                            elif np.argmax(sentiment) == 2:
                                category2_positive_count = category2_positive_count + 1

                        elif np.argmax(category) == 2:
                            category3 = category3 + 1
                            tweets_cat3.append(tweet.full_text)
                            stmnt_cat3.append('Customer Service')
                            if np.argmax(sentiment) == 0:
                                category3_negative_count = category3_negative_count + 1
                            elif np.argmax(sentiment) == 2:
                                category3_positive_count = category3_positive_count + 1

                        elif np.argmax(category) == 3:
                            category4 = category4 + 1
                            tweets_cat4.append(tweet.full_text)
                            stmnt_cat4.append('Other Matters')
                            if np.argmax(sentiment) == 0:
                                category4_negative_count = category4_negative_count + 1
                            elif np.argmax(sentiment) == 2:
                                category4_positive_count = category4_positive_count + 1

        except tweepy.TweepError:
            time.sleep(120)

    negoutput = dict(zip(neg_tweets, neg_stmnt))
    posoutput = dict(zip(pos_tweets, pos_stmnt))
    output_cat1 = dict(zip(tweets_cat1, stmnt_cat1))
    output_cat2 = dict(zip(tweets_cat2, stmnt_cat2))
    output_cat3 = dict(zip(tweets_cat3, stmnt_cat3))
    output_cat4 = dict(zip(tweets_cat4, stmnt_cat4))
    data = {'Task': 'Sentiment Analysis', 'Positive': positive_count, 'Negative': negative_count}
    Internet = {'Task': 'Internet', 'Positive': category1_positive_count, 'Negative': category1_negative_count}
    Network = {'Task': 'Network', 'Positive': category2_positive_count, 'Negative': category2_negative_count}
    Customer = {'Task': 'Customer', 'Positive': category3_positive_count, 'Negative': category3_negative_count}
    Other = {'Task': 'Other', 'Positive': category4_positive_count, 'Negative': category4_negative_count}
    return render_template('pie-chart-twitter.html', data=data, negoutputs=negoutput, posoutputs=posoutput,
                           output_cat1=output_cat1, output_cat2=output_cat2, output_cat3=output_cat3,
                           output_cat4=output_cat4, Internet=Internet, Network=Network, Customer=Customer, Other=Other,
                           tweetcnt=tweet_count, cat1=category1, cat2=category2, cat3=category3, cat4=category4)



@app.route('/user', methods=['GET'])
def user():
    return render_template('user.html')


# For User Data
# To use the predict button in the web-app
@app.route('/user', methods=['POST'])
def predict():
    cat1 = 0
    cat2 = 0
    cat3 = 0
    cat4 = 0
    poscnt = 0
    negcnt = 0
    tweetcnt = 0
    cat1_negcnt = 0
    cat1_poscnt = 0
    cat2_negcnt = 0
    cat2_poscnt = 0
    cat3_negcnt = 0
    cat3_poscnt = 0
    cat4_negcnt = 0
    cat4_poscnt = 0

    if request.method == "POST":
        if request.files:
            csv_upload = request.files["files"]
            # filename = csv_upload.filename
            csv_upload.save(os.path.join("uploads", csv_upload.filename))
            path = os.path.join("uploads", csv_upload.filename)
            n_tweets = []
            n_ops = []
            p_tweets = []
            p_ops = []
            tweets_cat1 = []
            ops_cat1 = []
            tweets_cat2 = []
            ops_cat2 = []
            tweets_cat3 = []
            ops_cat3 = []
            tweets_cat4 = []
            ops_cat4 = []
            with open(path, encoding='unicode_escape') as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    if (detect(row[0]) == 'en') :

                        a = preprocess_text(row[0])
                        print(a)
                        x = []
                        x.clear()
                        x.append(a)
                        w = tokenizer.texts_to_sequences(x)
                        # print(x)
                        # print(w)

                        twt1 = pad_sequences(w, maxlen=174, dtype='int32', value=0)

                        sentiment1 = sental_sentiment_analyzer_model.predict(twt1, batch_size=1, verbose=2)[0]
                        category = cat_model.predict(twt1)

                        # # sentiment
                        if np.argmax(sentiment1) == 0:
                            tweetcnt = tweetcnt + 1
                            negcnt = negcnt + 1
                            n_tweets.append(row[0])
                            n_ops.append('Negative')
                        elif np.argmax(sentiment1) == 2:
                            tweetcnt = tweetcnt + 1
                            poscnt = poscnt + 1
                            p_tweets.append(row[0])
                            p_ops.append('Positive')

                        # categories
                        if np.argmax(sentiment1) == 0 or np.argmax(sentiment1) == 2:
                            if np.argmax(category) == 0:
                                cat1 = cat1 + 1
                                tweets_cat1.append(row[0])
                                ops_cat1.append('4G/ Internet Homebroadband')
                                if np.argmax(sentiment1) == 0:
                                    cat1_negcnt = cat1_negcnt + 1
                                elif np.argmax(sentiment1) == 2:
                                    cat1_poscnt = cat1_poscnt + 1
                            elif np.argmax(category) == 1:
                                cat2 = cat2 + 1
                                tweets_cat2.append(row[0])
                                ops_cat2.append('Network/ Coverage')
                                if np.argmax(sentiment1) == 0:
                                    cat2_negcnt = cat2_negcnt + 1
                                elif np.argmax(sentiment1) == 2:
                                    cat2_poscnt = cat2_poscnt + 1
                            elif np.argmax(category) == 2:
                                cat3 = cat3 + 1
                                tweets_cat3.append(row[0])
                                ops_cat3.append('Customer Service')
                                if np.argmax(sentiment1) == 0:
                                    cat3_negcnt = cat3_negcnt + 1
                                elif np.argmax(sentiment1) == 2:
                                    cat3_poscnt = cat3_poscnt + 1
                            elif np.argmax(category) == 3:
                                cat4 = cat4 + 1
                                tweets_cat4.append(row[0])
                                ops_cat4.append('Other Matters')
                                if np.argmax(sentiment1) == 0:
                                    cat4_negcnt = cat4_negcnt + 1
                                elif np.argmax(sentiment1) == 2:
                                    cat4_poscnt = cat4_poscnt + 1

    negoutput = dict(zip(n_tweets, n_ops))
    posoutput = dict(zip(p_tweets, p_ops))
    output_cat1 = dict(zip(tweets_cat1, ops_cat1))
    output_cat2 = dict(zip(tweets_cat2, ops_cat2))
    output_cat3 = dict(zip(tweets_cat3, ops_cat3))
    output_cat4 = dict(zip(tweets_cat4, ops_cat4))
    data = {'Task': 'Sentiment Analysis', 'Positive': poscnt, 'Negative': negcnt}
    Internet = {'Task': 'Internet', 'Positive': cat1_poscnt, 'Negative': cat1_negcnt}
    Network = {'Task': 'Network', 'Positive': cat2_poscnt, 'Negative': cat2_negcnt}
    Customer = {'Task': 'Customer', 'Positive': cat3_poscnt, 'Negative': cat3_negcnt}
    Other = {'Task': 'Other', 'Positive': cat4_poscnt, 'Negative': cat4_negcnt}
    return render_template('pie-chart-user.html', data=data, negoutputs=negoutput, posoutputs=posoutput,
                           output_cat1=output_cat1, output_cat2=output_cat2, output_cat3=output_cat3,
                           output_cat4=output_cat4, Internet=Internet, Network=Network, Customer=Customer, Other=Other,
                           tweetcnt=tweetcnt, cat1=cat1, cat2=cat2, cat3=cat3, cat4=cat4)



if __name__ == "__main__":
    app.run(debug=True)
