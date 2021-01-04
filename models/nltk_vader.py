import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def setup_vader():
    nltk.download('vader_lexicon')


# Compound represents the emotional intensity ranging from -1(extremely negative) to 1(extremely positive)
def get_sentiment_vader(tweet):
    clean_tweet = str(tweet).strip()
    social_media_analyzer = SentimentIntensityAnalyzer()
    return social_media_analyzer.polarity_scores(clean_tweet)


# Scoring reference : https://github.com/cjhutto/vaderSentiment#about-the-scoring
def get_unimeasure_sentiment(sen_polarity):
    compound_score = sen_polarity['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif 0.05 > compound_score > -0.05:
        return "Neutral"
    else:
        return "Negative"

