import pickle
import flair
import stanza
from flask import Flask
from flask_restful import Resource, Api, reqparse
from models.nltk_sentiment_nb import remove_noise, download_packages, stopwords, word_tokenize
from models.nltk_vader import setup_vader, get_sentiment_vader, get_unimeasure_sentiment
from models.textblob_sentiment import get_sentiment_details

app = Flask(__name__)
api = Api(app)

# Setup NLTK and related packages(One time)
download_packages()
setup_vader()

# Request Parsing setup
parser = reqparse.RequestParser()
parser.add_argument('data', required=True, help="Need tweet data to perform sentiment analysis!")

# Twitter Sentiment NLTK Naive Bayes
nb_senti_file = open('models/tweet_senti_nb.pickle', 'rb')
nb_senti_classifier = pickle.load(nb_senti_file)
nb_senti_file.close()

# Flair load pretrained model
flair_pre_model = flair.models.TextClassifier.load('en-sentiment')

# Standford CORE NLP Stanza Sentiment Pipeline
stan_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,sentiment')


class SentimentNLTKNB(Resource):
    eng_stop_set = set(stopwords.words('english'))

    def post(self):
        args = parser.parse_args()
        tweet_data = str(args['data'])
        return {'tweet': str(tweet_data), 'sentiment': str(self.get_sentiment(tweet_data))}

    def get_sentiment(self, tweet):
        clean_tweet = self.get_tokenized_tweet(tweet)
        noise_clean_tokens = remove_noise(clean_tweet, self.eng_stop_set)
        nltk_token_tweets = self.get_nltk_tokens(noise_clean_tokens)
        return nb_senti_classifier.classify(nltk_token_tweets)

    def get_nltk_tokens(self, cleaned_tokens):
        return dict([token, True] for token in cleaned_tokens)

    def get_tokenized_tweet(self, tweet):
        return word_tokenize(tweet)


class SentimentNLTKVader(Resource):

    def post(self):
        args = parser.parse_args()
        tweet_data = str(args['data'])
        return {'tweet': str(tweet_data), 'sentiment_score': self.get_sentiment(tweet_data)}

    def get_sentiment(self, tweet):
        polarity_scores = get_sentiment_vader(tweet)
        uni_final_score = get_unimeasure_sentiment(polarity_scores)
        return {'overall_sentiment': str(uni_final_score), 'polarity_scores': polarity_scores}


class SentimentTextBlob(Resource):

    def post(self):
        args = parser.parse_args()
        tweet_data = str(args['data'])
        return {'tweet': str(tweet_data), 'sentiment_scores': self.get_sentiment(tweet_data)}

    def get_sentiment(self, tweet):
        return get_sentiment_details(tweet)


class SentimentFlair(Resource):
    def post(self):
        args = parser.parse_args()
        tweet_data = str(args['data'])
        return {'tweet': str(tweet_data), 'sentiment_scores': self.get_sentiment(tweet_data, flair_pre_model)}

    def get_cleaned_tweet(self, modelled_tweet):
        sentiment = modelled_tweet.labels
        sentiment_data_list = str(sentiment[0]).split()
        sentiment_data_dict = {'overall_sentiment': sentiment_data_list[0], 'polarity_score': sentiment_data_list[1]}
        return sentiment_data_dict

    def get_sentiment(self, tweet, model):
        modelled_tweet = flair.data.Sentence(tweet)
        model.predict(modelled_tweet)
        cleaned_sentiment_data = self.get_cleaned_tweet(modelled_tweet)
        return cleaned_sentiment_data

class SentimentStanfordCoreNLP(Resource):
    def post(self):
        args = parser.parse_args()
        tweet_data = str(args['data'])
        return {'tweet': str(tweet_data), 'sentiment_scores': self.get_sentiment(tweet_data, stan_pipeline)}

    def get_sentiment(self, tweet, model):
        doc = model(tweet)
        sentiment_data = doc.sentences[0].sentiment
        return sentiment_data

class SentimentAll(Resource):
    def post(self):
        args = parser.parse_args()
        tweet_data = str(args['data'])
        return {'tweet': str(tweet_data), 'sentiment_scores': self.get_sentiment(tweet_data, stan_pipeline, flair_pre_model)}

    def get_sentiment(self, tweet, stan_model, flair_model):
        stan_doc = stan_model(tweet)
        stan_sentiment_data = stan_doc.sentences[0].sentiment
        flair_modelled_tweet = flair.data.Sentence(tweet)
        flair_model.predict(flair_modelled_tweet)
        flair_sentiment_data = self.get_cleaned_tweet(flair_modelled_tweet)
        return {'stanford_core_nlp': stan_sentiment_data, 'flair': flair_sentiment_data}

    def get_cleaned_tweet(self, modelled_tweet):
        sentiment = modelled_tweet.labels
        sentiment_data_list = str(sentiment[0]).split()
        sentiment_data_dict = {'overall_sentiment': sentiment_data_list[0], 'polarity_score': sentiment_data_list[1]}
        return sentiment_data_dict


api.add_resource(SentimentNLTKNB, '/sentiment/nb')
api.add_resource(SentimentNLTKVader, '/sentiment/vader')
api.add_resource(SentimentTextBlob, '/sentiment/tb')
api.add_resource(SentimentFlair, '/sentiment/fl')
api.add_resource(SentimentStanfordCoreNLP, '/sentiment/scnlp')
api.add_resource(SentimentAll, '/sentiment/all')

if __name__ == '__main__':
    app.run(debug=True)
