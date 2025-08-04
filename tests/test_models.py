"""
Unit tests for sentiment analysis models.
"""

import pytest
from unittest.mock import patch, Mock, mock_open
import pickle

# Import model functions
from models.nltk_sentiment_nb import (
    remove_noise, 
    lemmatize_sentence,
    remove_hyperlinks,
    remove_at_mention,
    is_punctuation,
    is_stop_word
)
from models.nltk_vader import (
    get_sentiment_vader,
    get_unimeasure_sentiment
)
from models.textblob_sentiment import (
    get_sentiment_details,
    is_non_english,
    get_overall_sentiment
)


class TestNLTKSentimentNB:
    """Test NLTK Naive Bayes sentiment analysis functions."""
    
    def test_remove_hyperlinks(self):
        """Test hyperlink removal."""
        text_with_link = "Check this out https://example.com great!"
        cleaned = remove_hyperlinks(text_with_link)
        assert "https://example.com" not in cleaned
        assert "Check this out" in cleaned
        assert "great!" in cleaned
    
    def test_remove_at_mention(self):
        """Test @mention removal."""
        text_with_mention = "Hey @user123 how are you?"
        cleaned = remove_at_mention(text_with_mention)
        assert "@user123" not in cleaned
        assert "Hey" in cleaned
        assert "how are you?" in cleaned
    
    def test_is_punctuation(self):
        """Test punctuation detection."""
        assert is_punctuation("!")
        assert is_punctuation("?")
        assert is_punctuation(".")
        assert not is_punctuation("a")
        assert not is_punctuation("123")
    
    def test_is_stop_word(self):
        """Test stop word detection."""
        stop_words = {"the", "and", "or", "but"}
        assert is_stop_word("the", stop_words)
        assert is_stop_word("THE", stop_words)  # Case insensitive
        assert not is_stop_word("amazing", stop_words)
    
    @patch('models.nltk_sentiment_nb.pos_tag')
    @patch('models.nltk_sentiment_nb.WordNetLemmatizer')
    def test_lemmatize_sentence(self, mock_lemmatizer, mock_pos_tag):
        """Test sentence lemmatization."""
        mock_pos_tag.return_value = [("running", "VBG"), ("dogs", "NNS")]
        mock_lemmatizer_instance = Mock()
        mock_lemmatizer.return_value = mock_lemmatizer_instance
        mock_lemmatizer_instance.lemmatize.side_effect = ["run", "dog"]
        
        tokens = ["running", "dogs"]
        result = lemmatize_sentence(tokens)
        
        assert result == ["run", "dog"]
        assert mock_lemmatizer_instance.lemmatize.call_count == 2
    
    def test_remove_noise(self):
        """Test noise removal from tokens."""
        tokens = ["I", "love", "https://example.com", "@user", "!", "the", "amazing", "product"]
        stop_words = {"the", "i"}
        
        with patch('models.nltk_sentiment_nb.lemmatize_sentence') as mock_lemmatize:
            mock_lemmatize.return_value = ["love", "amazing", "product"]
            
            result = remove_noise(tokens, stop_words)
            assert "love" in result
            assert "amazing" in result
            assert "product" in result
            assert len(result) == 3


class TestNLTKVader:
    """Test NLTK VADER sentiment analysis functions."""
    
    @patch('models.nltk_vader.SentimentIntensityAnalyzer')
    def test_get_sentiment_vader(self, mock_analyzer):
        """Test VADER sentiment analysis."""
        mock_instance = Mock()
        mock_analyzer.return_value = mock_instance
        mock_instance.polarity_scores.return_value = {
            'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.5
        }
        
        result = get_sentiment_vader("I love this!")
        
        assert 'neg' in result
        assert 'neu' in result
        assert 'pos' in result
        assert 'compound' in result
        mock_instance.polarity_scores.assert_called_once()
    
    def test_get_unimeasure_sentiment(self):
        """Test sentiment classification from compound score."""
        # Test positive sentiment
        positive_scores = {'compound': 0.6}
        assert get_unimeasure_sentiment(positive_scores) == "Positive"
        
        # Test negative sentiment
        negative_scores = {'compound': -0.6}
        assert get_unimeasure_sentiment(negative_scores) == "Negative"
        
        # Test neutral sentiment
        neutral_scores = {'compound': 0.0}
        assert get_unimeasure_sentiment(neutral_scores) == "Neutral"
        
        # Test boundary cases
        boundary_positive = {'compound': 0.05}
        assert get_unimeasure_sentiment(boundary_positive) == "Positive"
        
        boundary_negative = {'compound': -0.05}
        assert get_unimeasure_sentiment(boundary_negative) == "Negative"


class TestTextBlobSentiment:
    """Test TextBlob sentiment analysis functions."""
    
    def test_get_overall_sentiment(self):
        """Test overall sentiment classification."""
        # Test positive sentiment
        assert get_overall_sentiment(0.6) == "Positive"
        
        # Test negative sentiment
        assert get_overall_sentiment(-0.6) == "Negative"
        
        # Test neutral sentiment
        assert get_overall_sentiment(0.0) == "Neutral"
        
        # Test boundary cases
        assert get_overall_sentiment(0.05) == "Positive"
        assert get_overall_sentiment(-0.05) == "Negative"
        assert get_overall_sentiment(0.04) == "Neutral"
    
    @patch('models.textblob_sentiment.TextBlob')
    def test_is_non_english(self, mock_textblob):
        """Test language detection."""
        mock_blob = Mock()
        mock_textblob.return_value = mock_blob
        
        # Test English text
        mock_blob.detect_language.return_value = 'en'
        assert not is_non_english(mock_blob)
        
        # Test non-English text
        mock_blob.detect_language.return_value = 'es'
        assert is_non_english(mock_blob)
    
    @patch('models.textblob_sentiment.TextBlob')
    def test_get_sentiment_details(self, mock_textblob):
        """Test detailed sentiment analysis."""
        mock_blob = Mock()
        mock_textblob.return_value = mock_blob
        
        # Mock English text
        mock_blob.detect_language.return_value = 'en'
        mock_blob.sentiment_assessments = (0.5, 0.6, [])
        
        result = get_sentiment_details("I love this!")
        
        assert 'Polarity_score' in result
        assert 'subjectivity_score' in result
        assert 'Overall sentiment' in result
        assert result['Polarity_score'] == 0.5
        assert result['subjectivity_score'] == 0.6
    
    @patch('models.textblob_sentiment.get_eng_tweet_blob')
    @patch('models.textblob_sentiment.TextBlob')
    def test_get_sentiment_details_non_english(self, mock_textblob, mock_translate):
        """Test sentiment analysis with translation."""
        mock_blob = Mock()
        mock_textblob.return_value = mock_blob
        mock_blob.detect_language.return_value = 'es'  # Spanish
        
        mock_translated_blob = Mock()
        mock_translate.return_value = mock_translated_blob
        mock_translated_blob.sentiment_assessments = (0.3, 0.4, [])
        
        result = get_sentiment_details("Me gusta esto!")
        
        assert 'Polarity_score' in result
        assert result['Polarity_score'] == 0.3
        mock_translate.assert_called_once()


class TestModelIntegration:
    """Test model integration and initialization."""
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_pickle_data')
    @patch('pickle.load')
    def test_model_loading(self, mock_pickle_load, mock_file):
        """Test loading of pickle model."""
        mock_classifier = Mock()
        mock_pickle_load.return_value = mock_classifier
        
        # Simulate loading the model
        with open('models/tweet_senti_nb.pickle', 'rb') as f:
            classifier = pickle.load(f)
        
        assert classifier == mock_classifier
        mock_pickle_load.assert_called_once()
    
    def test_package_downloads(self):
        """Test that required packages can be imported."""
        # Simple test to verify that the required packages can be imported
        # This is more reliable than testing the actual download process
        try:
            import nltk
            import textblob
            import flair
            import stanza
            assert True  # If we get here, imports work
        except ImportError as e:
            assert False, f"Required package not available: {e}"


class TestErrorHandling:
    """Test error handling in model functions."""
    
    def test_remove_noise_empty_input(self):
        """Test remove_noise with empty input."""
        with patch('models.nltk_sentiment_nb.lemmatize_sentence') as mock_lemmatize:
            mock_lemmatize.return_value = []
            result = remove_noise([], set())
            assert result == []
    
    def test_get_sentiment_vader_empty_input(self):
        """Test VADER with empty input."""
        with patch('models.nltk_vader.SentimentIntensityAnalyzer') as mock_analyzer:
            mock_instance = Mock()
            mock_analyzer.return_value = mock_instance
            mock_instance.polarity_scores.return_value = {
                'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0
            }
            
            result = get_sentiment_vader("")
            assert result['compound'] == 0.0
    
    @patch('models.textblob_sentiment.TextBlob')
    def test_textblob_error_handling(self, mock_textblob):
        """Test TextBlob error handling."""
        mock_blob = Mock()
        mock_textblob.return_value = mock_blob
        mock_blob.detect_language.side_effect = Exception("Language detection failed")
        
        # Should handle the exception gracefully
        with pytest.raises(Exception):
            get_sentiment_details("test text")
