#!/usr/bin/env python3
"""
Sentiment Analysis API

A comprehensive Flask-based REST API for sentiment analysis using multiple
state-of-the-art NLP libraries and models.

Author: Gajesh Bhat
Version: 2.0.0
License: MIT
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

import flair
import stanza
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse

from werkzeug.exceptions import BadRequest, InternalServerError

from models.nltk_sentiment_nb import (
    remove_noise,
    download_packages,
    stopwords,
    word_tokenize
)
from models.nltk_vader import (
    setup_vader,
    get_sentiment_vader,
    get_unimeasure_sentiment
)
from models.textblob_sentiment import get_sentiment_details

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Configuration
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
app.config['HOST'] = os.getenv('FLASK_HOST', '127.0.0.1')
app.config['PORT'] = int(os.getenv('FLASK_PORT', 5000))

# Global variables for models
nb_senti_classifier = None
flair_pre_model = None
stan_pipeline = None

def initialize_models() -> None:
    """
    Initialize all sentiment analysis models.

    This function sets up NLTK packages, loads pre-trained models,
    and initializes the sentiment analysis pipelines.

    Raises:
        Exception: If model initialization fails
    """
    global nb_senti_classifier, flair_pre_model, stan_pipeline

    try:
        logger.info("Initializing sentiment analysis models...")

        # Setup NLTK and related packages
        logger.info("Downloading NLTK packages...")
        download_packages()
        setup_vader()

        # Load NLTK Naive Bayes classifier
        logger.info("Loading NLTK Naive Bayes classifier...")
        model_path = Path('models/tweet_senti_nb.pickle')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as nb_file:
            nb_senti_classifier = pickle.load(nb_file)

        # Load Flair pre-trained model
        logger.info("Loading Flair sentiment model...")
        try:
            # Set environment variable to handle PyTorch 2.6+ compatibility
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

            # Monkey patch torch.load to handle weights_only parameter
            import torch
            original_load = torch.load

            def patched_load(*args, **kwargs):
                # Remove weights_only if it's causing issues, or set it to False
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)

            torch.load = patched_load

            try:
                flair_pre_model = flair.models.TextClassifier.load('en-sentiment')
                logger.info("Flair model loaded successfully!")
            finally:
                # Restore original torch.load
                torch.load = original_load

        except Exception as e:
            logger.error(f"Failed to load Flair model: {str(e)}")
            logger.warning("Continuing without Flair model - /sentiment/fl and /sentiment/all endpoints will not work")
            flair_pre_model = None

        # Initialize Stanford CoreNLP Stanza pipeline
        logger.info("Initializing Stanza pipeline...")
        try:
            # Apply the same PyTorch compatibility fix for Stanza
            import torch
            original_load = torch.load

            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)

            torch.load = patched_load

            try:
                stan_pipeline = stanza.Pipeline(
                    lang='en',
                    processors='tokenize,mwt,pos,lemma,sentiment',
                    verbose=False
                )
                logger.info("Stanza pipeline initialized successfully!")
            finally:
                # Restore original torch.load
                torch.load = original_load

        except Exception as e:
            logger.error(f"Failed to initialize Stanza pipeline: {str(e)}")
            logger.warning("Continuing without Stanza pipeline - /sentiment/scnlp and /sentiment/all endpoints will not work")
            stan_pipeline = None

        logger.info("All models initialized successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise

# Request parser setup
parser = reqparse.RequestParser()
parser.add_argument(
    'data',
    required=True,
    help="Text data is required for sentiment analysis!",
    type=str
)


class SentimentNLTKNB(Resource):
    """
    NLTK Naive Bayes sentiment analysis resource.

    Uses a pre-trained Naive Bayes classifier with ~99% accuracy on tweet data.
    Performs text preprocessing including tokenization, noise removal, and lemmatization.
    """

    def __init__(self):
        """Initialize the resource with English stopwords."""
        self.eng_stop_set = set(stopwords.words('english'))

    def post(self) -> Dict[str, Any]:
        """
        Analyze sentiment using NLTK Naive Bayes classifier.

        Returns:
            Dict containing the original text and predicted sentiment

        Raises:
            BadRequest: If input data is invalid
            InternalServerError: If sentiment analysis fails
        """
        try:
            args = parser.parse_args()
            tweet_data = str(args['data']).strip()

            if not tweet_data:
                raise BadRequest("Input text cannot be empty")

            sentiment = self.get_sentiment(tweet_data)

            return {
                'tweet': tweet_data,
                'sentiment': sentiment,
                'model': 'NLTK Naive Bayes',
                'confidence': 'High (~99% accuracy)'
            }

        except BadRequest:
            raise
        except Exception as e:
            logger.error(f"NLTK NB sentiment analysis failed: {str(e)}")
            raise InternalServerError("Sentiment analysis failed")

    def get_sentiment(self, tweet: str) -> str:
        """
        Get sentiment prediction for the given text.

        Args:
            tweet: Input text to analyze

        Returns:
            Sentiment prediction ('Positive' or 'Negative')
        """
        clean_tweet = self.get_tokenized_tweet(tweet)
        noise_clean_tokens = remove_noise(clean_tweet, self.eng_stop_set)
        nltk_token_tweets = self.get_nltk_tokens(noise_clean_tokens)
        return nb_senti_classifier.classify(nltk_token_tweets)

    def get_nltk_tokens(self, cleaned_tokens: list) -> Dict[str, bool]:
        """
        Convert cleaned tokens to NLTK format.

        Args:
            cleaned_tokens: List of preprocessed tokens

        Returns:
            Dictionary with tokens as keys and True as values
        """
        return dict([token, True] for token in cleaned_tokens)

    def get_tokenized_tweet(self, tweet: str) -> list:
        """
        Tokenize the input text.

        Args:
            tweet: Input text to tokenize

        Returns:
            List of tokens
        """
        return word_tokenize(tweet)


class SentimentNLTKVader(Resource):
    """
    NLTK VADER sentiment analysis resource.

    Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis.
    Particularly effective for social media text, handles emoticons, slang, and punctuation.
    """

    def post(self) -> Dict[str, Any]:
        """
        Analyze sentiment using NLTK VADER.

        Returns:
            Dict containing the original text and detailed sentiment scores

        Raises:
            BadRequest: If input data is invalid
            InternalServerError: If sentiment analysis fails
        """
        try:
            args = parser.parse_args()
            tweet_data = str(args['data']).strip()

            if not tweet_data:
                raise BadRequest("Input text cannot be empty")

            sentiment_data = self.get_sentiment(tweet_data)

            return {
                'tweet': tweet_data,
                'sentiment_score': sentiment_data,
                'model': 'NLTK VADER',
                'description': 'Lexicon-based sentiment analysis'
            }

        except BadRequest:
            raise
        except Exception as e:
            logger.error(f"NLTK VADER sentiment analysis failed: {str(e)}")
            raise InternalServerError("Sentiment analysis failed")

    def get_sentiment(self, tweet: str) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Get VADER sentiment scores for the given text.

        Args:
            tweet: Input text to analyze

        Returns:
            Dictionary containing overall sentiment and detailed polarity scores
        """
        polarity_scores = get_sentiment_vader(tweet)
        uni_final_score = get_unimeasure_sentiment(polarity_scores)
        return {
            'overall_sentiment': str(uni_final_score),
            'polarity_scores': polarity_scores
        }


class SentimentTextBlob(Resource):
    """
    TextBlob sentiment analysis resource.

    Uses TextBlob for sentiment analysis with automatic language detection
    and translation capabilities. Provides polarity and subjectivity scores.
    """

    def post(self) -> Dict[str, Any]:
        """
        Analyze sentiment using TextBlob.

        Returns:
            Dict containing the original text and detailed sentiment analysis

        Raises:
            BadRequest: If input data is invalid
            InternalServerError: If sentiment analysis fails
        """
        try:
            args = parser.parse_args()
            tweet_data = str(args['data']).strip()

            if not tweet_data:
                raise BadRequest("Input text cannot be empty")

            sentiment_data = self.get_sentiment(tweet_data)

            return {
                'tweet': tweet_data,
                'sentiment_scores': sentiment_data,
                'model': 'TextBlob',
                'description': 'Polarity and subjectivity analysis with language detection'
            }

        except BadRequest:
            raise
        except Exception as e:
            logger.error(f"TextBlob sentiment analysis failed: {str(e)}")
            raise InternalServerError("Sentiment analysis failed")

    def get_sentiment(self, tweet: str) -> Dict[str, Any]:
        """
        Get TextBlob sentiment analysis for the given text.

        Args:
            tweet: Input text to analyze

        Returns:
            Dictionary containing polarity, subjectivity, and overall sentiment
        """
        return get_sentiment_details(tweet)


class SentimentFlair(Resource):
    """
    Flair sentiment analysis resource.

    Uses Flair pre-trained sentiment model for high-accuracy sentiment analysis.
    """

    def post(self) -> Dict[str, Any]:
        """
        Analyze sentiment using Flair.

        Returns:
            Dict containing the original text and sentiment analysis results

        Raises:
            BadRequest: If input data is invalid
            InternalServerError: If sentiment analysis fails
        """
        try:
            args = parser.parse_args()
            tweet_data = str(args['data']).strip()

            if not tweet_data:
                raise BadRequest("Input text cannot be empty")

            sentiment_data = self.get_sentiment(tweet_data, flair_pre_model)

            return {
                'tweet': tweet_data,
                'sentiment_scores': sentiment_data,
                'model': 'Flair',
                'description': 'Deep learning-based sentiment analysis'
            }

        except BadRequest:
            raise
        except Exception as e:
            logger.error(f"Flair sentiment analysis failed: {str(e)}")
            raise InternalServerError("Sentiment analysis failed")

    def get_cleaned_tweet(self, modelled_tweet):
        sentiment = modelled_tweet.labels
        if sentiment:
            # Extract sentiment label and confidence score
            label = sentiment[0]
            sentiment_value = label.value  # e.g., 'POSITIVE' or 'NEGATIVE'
            confidence_score = label.score  # confidence score

            sentiment_data_dict = {
                'overall_sentiment': sentiment_value,
                'polarity_score': f"{confidence_score:.4f}"
            }
            return sentiment_data_dict
        else:
            return {'overall_sentiment': 'NEUTRAL', 'polarity_score': '0.0000'}

    def get_sentiment(self, tweet, model):
        if model is None:
            raise Exception("Flair model is not available")

        modelled_tweet = flair.data.Sentence(tweet)
        model.predict(modelled_tweet)
        cleaned_sentiment_data = self.get_cleaned_tweet(modelled_tweet)
        return cleaned_sentiment_data

class SentimentStanfordCoreNLP(Resource):
    """
    Stanford CoreNLP sentiment analysis resource.

    Uses Stanza pipeline for sentiment analysis with neural network models.
    """

    def post(self) -> Dict[str, Any]:
        """
        Analyze sentiment using Stanford CoreNLP.

        Returns:
            Dict containing the original text and sentiment score

        Raises:
            BadRequest: If input data is invalid
            InternalServerError: If sentiment analysis fails
        """
        try:
            args = parser.parse_args()
            tweet_data = str(args['data']).strip()

            if not tweet_data:
                raise BadRequest("Input text cannot be empty")

            sentiment_data = self.get_sentiment(tweet_data, stan_pipeline)

            return {
                'tweet': tweet_data,
                'sentiment_scores': sentiment_data,
                'model': 'Stanford CoreNLP',
                'description': 'Neural network-based sentiment analysis'
            }

        except BadRequest:
            raise
        except Exception as e:
            logger.error(f"Stanford CoreNLP sentiment analysis failed: {str(e)}")
            raise InternalServerError("Sentiment analysis failed")

    def get_sentiment(self, tweet, model):
        if model is None:
            raise Exception("Stanford CoreNLP model is not available")

        doc = model(tweet)
        sentiment_data = doc.sentences[0].sentiment
        return sentiment_data

class SentimentAll(Resource):
    """
    Combined sentiment analysis resource.

    Uses both Stanford CoreNLP and Flair models for comprehensive analysis.
    """

    def post(self) -> Dict[str, Any]:
        """
        Analyze sentiment using multiple models.

        Returns:
            Dict containing the original text and combined sentiment analysis

        Raises:
            BadRequest: If input data is invalid
            InternalServerError: If sentiment analysis fails
        """
        try:
            args = parser.parse_args()
            tweet_data = str(args['data']).strip()

            if not tweet_data:
                raise BadRequest("Input text cannot be empty")

            sentiment_data = self.get_sentiment(tweet_data, stan_pipeline, flair_pre_model)

            return {
                'tweet': tweet_data,
                'sentiment_scores': sentiment_data,
                'model': 'Combined (Stanford CoreNLP + Flair)',
                'description': 'Multi-model sentiment analysis'
            }

        except BadRequest:
            raise
        except Exception as e:
            logger.error(f"Combined sentiment analysis failed: {str(e)}")
            raise InternalServerError("Sentiment analysis failed")

    def get_sentiment(self, tweet, stan_model, flair_model):
        result = {}

        # Handle Stanford CoreNLP
        if stan_model is not None:
            try:
                stan_doc = stan_model(tweet)
                stan_sentiment_data = stan_doc.sentences[0].sentiment
                result['stanford_core_nlp'] = stan_sentiment_data
            except Exception as e:
                logger.warning(f"Stanford CoreNLP analysis failed: {str(e)}")
                result['stanford_core_nlp'] = {'error': 'Stanford CoreNLP model not available'}
        else:
            result['stanford_core_nlp'] = {'error': 'Stanford CoreNLP model not available'}

        # Handle Flair
        if flair_model is not None:
            try:
                flair_modelled_tweet = flair.data.Sentence(tweet)
                flair_model.predict(flair_modelled_tweet)
                flair_sentiment_data = self.get_cleaned_tweet(flair_modelled_tweet)
                result['flair'] = flair_sentiment_data
            except Exception as e:
                logger.warning(f"Flair analysis failed: {str(e)}")
                result['flair'] = {'error': 'Flair model not available'}
        else:
            result['flair'] = {'error': 'Flair model not available'}

        return result

    def get_cleaned_tweet(self, modelled_tweet):
        sentiment = modelled_tweet.labels
        if sentiment:
            # Extract sentiment label and confidence score
            label = sentiment[0]
            sentiment_value = label.value  # e.g., 'POSITIVE' or 'NEGATIVE'
            confidence_score = label.score  # confidence score

            sentiment_data_dict = {
                'overall_sentiment': sentiment_value,
                'polarity_score': f"{confidence_score:.4f}"
            }
            return sentiment_data_dict
        else:
            return {'overall_sentiment': 'NEUTRAL', 'polarity_score': '0.0000'}


# Error handlers
@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({
        'error': 'Bad Request',
        'message': str(error.description),
        'status_code': 400
    }), 400

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'status_code': 500
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Sentiment Analysis API is running',
        'version': '2.0.0'
    })

@app.route('/', methods=['GET'])
def index():
    """API information endpoint."""
    return jsonify({
        'name': 'Sentiment Analysis API',
        'version': '2.0.0',
        'description': 'A comprehensive REST API for sentiment analysis using multiple NLP libraries',
        'endpoints': {
            '/sentiment/nb': 'NLTK Naive Bayes sentiment analysis',
            '/sentiment/vader': 'NLTK VADER sentiment analysis',
            '/sentiment/tb': 'TextBlob sentiment analysis',
            '/sentiment/fl': 'Flair sentiment analysis',
            '/sentiment/scnlp': 'Stanford CoreNLP sentiment analysis',
            '/sentiment/all': 'Combined sentiment analysis (Stanford + Flair)',
            '/health': 'Health check endpoint'
        },
        'usage': 'Send POST requests with JSON payload: {"data": "text to analyze"}'
    })

# Register API resources
api.add_resource(SentimentNLTKNB, '/sentiment/nb')
api.add_resource(SentimentNLTKVader, '/sentiment/vader')
api.add_resource(SentimentTextBlob, '/sentiment/tb')
api.add_resource(SentimentFlair, '/sentiment/fl')
api.add_resource(SentimentStanfordCoreNLP, '/sentiment/scnlp')
api.add_resource(SentimentAll, '/sentiment/all')

def main():
    """
    Main function to run the Flask application.

    Initializes models and starts the Flask development server.
    """
    try:
        # Initialize all models
        initialize_models()

        # Start the Flask application
        logger.info(f"Starting Sentiment Analysis API on {app.config['HOST']}:{app.config['PORT']}")
        app.run(
            host=app.config['HOST'],
            port=app.config['PORT'],
            debug=app.config['DEBUG']
        )

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
