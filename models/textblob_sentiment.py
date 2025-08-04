from textblob import TextBlob


# Install Textblob corpus : Goes in setups script : python -m textblob.download_corpora

def is_non_english(tweet_blob):
    try:
        # Try to detect language if the method exists
        if hasattr(tweet_blob, 'detect_language'):
            tweet_lang = tweet_blob.detect_language()
            if str(tweet_lang) != 'en':
                return True
        # If language detection is not available, assume English
        return False
    except Exception:
        # If language detection fails, assume English
        return False


def get_eng_tweet_blob(tweet_blob):
    return TextBlob(tweet_blob.translate(to='en'))


# Scoring reference : https://github.com/cjhutto/vaderSentiment#about-the-scoring
def get_overall_sentiment(polarity_score):
    if polarity_score >= 0.05:
        return "Positive"
    elif 0.05 > polarity_score > -0.05:
        return "Neutral"
    else:
        return "Negative"


# score < 0.5 more objective and score more that 0.5 is subjective and opinion based. Useful for News related tweets
def get_sentiment_details(tweet):
    tweet_blob = TextBlob(str(tweet).strip())
    if is_non_english(tweet_blob) is True:
        tweet_blob = get_eng_tweet_blob(tweet_blob)
    tweet_sentiment = tweet_blob.sentiment_assessments
    sentiment_results = {
        'Polarity_score': tweet_sentiment[0],
        'subjectivity_score': tweet_sentiment[1],
        'assessment_details': tweet_sentiment[2],
        'Overall sentiment': get_overall_sentiment(tweet_sentiment[0])
    }
    return sentiment_results
