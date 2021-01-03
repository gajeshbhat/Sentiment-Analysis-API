import re
import string
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts.
    DEFAULTS to Noun. Uses the first letter to cover various forms of POS"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_sentence(tokens):
    tweet_lemming = WordNetLemmatizer()
    tweet_lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        lem_pos = get_wordnet_pos(tag[0])
        tweet_lemmatized_sentence.append(tweet_lemming.lemmatize(word, lem_pos))
    return tweet_lemmatized_sentence


def remove_hyperlinks(token):
    clean_token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                         '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
    return clean_token


def remove_at_mention(token):
    return re.sub("(@[A-Za-z0-9_]+)", "", token)


def is_punctuation(token):
    return token in string.punctuation


def is_stop_word(token, stop_words):
    return token.lower() in stop_words


def remove_noise(tokens, stop_words):
    # Clean Hyperlink, @mentions and stop words and punctuations
    clean_tokens = list()
    for i in range(0, len(tokens)):
        current_token = tokens[i]
        hyper_clean_token = remove_hyperlinks(current_token)
        mention_cleaned_token = remove_at_mention(hyper_clean_token)  # Fully clean
        if len(mention_cleaned_token) > 0 and not is_punctuation(mention_cleaned_token) and not is_stop_word(
                mention_cleaned_token, stop_words):
            clean_tokens.append(mention_cleaned_token)
    # Lemmatization
    clean_lem_tokens = lemmatize_sentence(clean_tokens)
    return clean_lem_tokens


def download_packages():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('punkt')
