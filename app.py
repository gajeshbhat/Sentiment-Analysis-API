import pickle
from flask import Flask
from flask_restful import Resource, Api, reqparse
from models.nltk_sentiment_nb import remove_noise, download_packages, stopwords, word_tokenize


app = Flask(__name__)
api = Api(app)

# Setup NLTK (One time)
download_packages()

# Request Parsing setup
parser = reqparse.RequestParser()
parser.add_argument('data', required=True, help="Need tweet data to perform sentiment analysis!")

# Twitter Sentiment NLTK Naive Bayes
nb_senti_file = open('models/tweet_senti_nb.pickle', 'rb')
nb_senti_classifier = pickle.load(nb_senti_file)
nb_senti_file.close()


class SentimentNLTKNB(Resource):

    eng_stop_set = set(stopwords.words('english'))

    def post(self):
        args = parser.parse_args()
        tweet_data = str(args['data'])
        return {'tweet': str(tweet_data),'sentiment': str(self.get_sentiment(tweet_data))}

    def get_sentiment(self,tweet):
        clean_tweet = self.get_tokenized_tweet(tweet)
        noise_clean_tokens = remove_noise(clean_tweet,self.eng_stop_set)
        nltk_token_tweets = self.get_nltk_tokens(noise_clean_tokens)
        return nb_senti_classifier.classify(nltk_token_tweets)

    def get_nltk_tokens(self, cleaned_tokens):
        return dict([token, True] for token in cleaned_tokens)

    def get_tokenized_tweet(self, tweet):
        return word_tokenize(tweet)


api.add_resource(SentimentNLTKNB, '/sentiment')

if __name__ == '__main__':
    app.run(debug=True)