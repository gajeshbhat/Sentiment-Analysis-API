import pickle
from flask import Flask
from flask_restful import Resource, Api
from nltk.tokenize import word_tokenize

app = Flask(__name__)
api = Api(app)

# Twitter Sentiment NLTK Naive Bayes
nb_senti_file = open('models/tweet_senti_nb.pickle', 'rb')
nb_senti_classifier = pickle.load(nb_senti_file)
nb_senti_file.close()


class SentimentNLTKNB(Resource):
    def get(self):
        return {'sentiment': str(self.get_sentiment())}

    def get_sentiment(self,tweet):
        clean_tweet = self.get_tokenized_tweet(tweet)
        tokenized_tweet = self.get_tokenized_tweet(clean_tweet)
        return nb_senti_classifier.classify(tokenized_tweet)

    def get_nltk_tokens(self, cleaned_tokens):
        return dict([token, True] for token in cleaned_tokens)

    def get_tokenized_tweet(self, tweet):
        return word_tokenize(tweet)


api.add_resource(SentimentNLTKNB, '/')

if __name__ == '__main__':
    app.run(debug=True)
