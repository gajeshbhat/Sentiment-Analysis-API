"""
Pytest configuration and fixtures for Sentiment Analysis API tests.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def setup_nltk_data():
    """Download required NLTK data for tests, but skip in CI if not available."""
    import nltk
    import os

    # Skip NLTK downloads in CI environment to avoid timeouts
    if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
        return

    nltk_packages = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('corpora/wordnet', 'wordnet'),
        ('sentiment/vader_lexicon', 'vader_lexicon'),
    ]

    for path, package in nltk_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                # Skip if download fails (e.g., in CI)
                pass

@pytest.fixture
def mock_models():
    """Mock all the heavy NLP models to speed up testing."""
    with patch('app.initialize_models') as mock_init:
        # Mock the global model variables
        with patch('app.nb_senti_classifier') as mock_nb, \
             patch('app.flair_pre_model') as mock_flair, \
             patch('app.stan_pipeline') as mock_stanza:

            # Configure mock behaviors
            mock_nb.classify.return_value = "Positive"
            mock_flair.predict = Mock()
            mock_stanza.return_value = Mock()

            yield {
                'nb_classifier': mock_nb,
                'flair_model': mock_flair,
                'stanza_pipeline': mock_stanza,
                'init_models': mock_init
            }

@pytest.fixture
def app():
    """Create and configure a test Flask application."""
    # Mock all heavy dependencies before importing app
    with patch('flair.models.TextClassifier.load') as mock_flair_load, \
         patch('stanza.Pipeline') as mock_stanza, \
         patch('app.initialize_models') as mock_init, \
         patch('builtins.open'), \
         patch('pickle.load') as mock_pickle:

        # Configure mocks
        mock_flair_load.return_value = Mock()
        mock_stanza.return_value = Mock()
        mock_pickle.return_value = Mock()
        mock_init.return_value = None

        from app import app as flask_app

        # Configure the app for testing
        flask_app.config['TESTING'] = True
        flask_app.config['DEBUG'] = False

        # Mock the global model variables
        import app
        app.nb_senti_classifier = Mock()
        app.nb_senti_classifier.classify.return_value = "Positive"
        app.flair_pre_model = Mock()
        app.stan_pipeline = Mock()

        return flask_app

@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()

@pytest.fixture
def sample_texts():
    """Sample texts for testing sentiment analysis."""
    return {
        'positive': [
            "I love this amazing product!",
            "This is fantastic and wonderful!",
            "Great job, excellent work!",
            "I'm so happy and excited!",
            "Best day ever! 😊"
        ],
        'negative': [
            "I hate this terrible product!",
            "This is awful and horrible!",
            "Worst experience ever!",
            "I'm so angry and frustrated!",
            "Terrible service! 😠"
        ],
        'neutral': [
            "This is a product.",
            "The weather is okay.",
            "It's a normal day.",
            "The meeting is at 3 PM.",
            "Here is some information."
        ],
        'empty': [
            "",
            "   ",
            "\n\t",
        ],
        'special': [
            "Check out this link: https://example.com",
            "@user mentioned something interesting",
            "Price is $99.99 #sale #discount",
            "Mixed emotions 😊😢",
            "Text with numbers 123 and symbols @#$%"
        ]
    }

@pytest.fixture
def api_endpoints():
    """List of API endpoints to test."""
    return [
        '/sentiment/nb',
        '/sentiment/vader',
        '/sentiment/tb',
        '/sentiment/fl',
        '/sentiment/scnlp',
        '/sentiment/all'
    ]

@pytest.fixture
def expected_response_keys():
    """Expected keys in API responses for different endpoints."""
    return {
        '/sentiment/nb': ['tweet', 'sentiment', 'model'],
        '/sentiment/vader': ['tweet', 'sentiment_score', 'model'],
        '/sentiment/tb': ['tweet', 'sentiment_scores', 'model'],
        '/sentiment/fl': ['tweet', 'sentiment_scores', 'model'],
        '/sentiment/scnlp': ['tweet', 'sentiment_scores', 'model'],
        '/sentiment/all': ['tweet', 'sentiment_scores', 'model']
    }
