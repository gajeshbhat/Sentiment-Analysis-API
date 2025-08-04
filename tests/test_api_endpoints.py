"""
Integration tests for API endpoints.
"""

import json
import pytest
from unittest.mock import patch, Mock


class TestAPIEndpoints:
    """Test all API endpoints."""
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'message' in data
    
    def test_index_endpoint(self, client):
        """Test the index endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'name' in data
        assert 'version' in data
        assert 'endpoints' in data
        assert 'usage' in data
    
    @patch('models.nltk_sentiment_nb.remove_noise')
    @patch('models.nltk_sentiment_nb.word_tokenize')
    @patch('app.nb_senti_classifier')
    def test_nltk_nb_endpoint(self, mock_classifier, mock_tokenize, mock_remove_noise, client, sample_texts):
        """Test NLTK Naive Bayes endpoint."""
        mock_classifier.classify.return_value = "Positive"
        mock_tokenize.return_value = ['test', 'tokens']
        mock_remove_noise.return_value = ['clean', 'tokens']

        for text in sample_texts['positive']:
            response = client.post('/sentiment/nb',
                                 json={'data': text})
            assert response.status_code == 200

            data = json.loads(response.data)
            assert 'tweet' in data
            assert 'sentiment' in data
            assert 'model' in data
            assert data['tweet'] == text
    
    @patch('models.nltk_vader.get_sentiment_vader')
    @patch('models.nltk_vader.get_unimeasure_sentiment')
    def test_vader_endpoint(self, mock_unimeasure, mock_vader, client, sample_texts):
        """Test NLTK VADER endpoint."""
        mock_vader.return_value = {
            'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.5
        }
        mock_unimeasure.return_value = "Positive"
        
        for text in sample_texts['positive']:
            response = client.post('/sentiment/vader', 
                                 json={'data': text})
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'tweet' in data
            assert 'sentiment_score' in data
            assert 'model' in data
    
    @patch('models.textblob_sentiment.get_sentiment_details')
    def test_textblob_endpoint(self, mock_textblob, client, sample_texts):
        """Test TextBlob endpoint."""
        mock_textblob.return_value = {
            'Polarity_score': 0.5,
            'subjectivity_score': 0.6,
            'Overall sentiment': 'Positive'
        }
        
        for text in sample_texts['positive']:
            response = client.post('/sentiment/tb', 
                                 json={'data': text})
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'tweet' in data
            assert 'sentiment_scores' in data
            assert 'model' in data
    
    def test_missing_data_parameter(self, client, api_endpoints):
        """Test endpoints with missing data parameter."""
        for endpoint in api_endpoints:
            response = client.post(endpoint, json={})
            assert response.status_code == 400
    
    def test_empty_data_parameter(self, client, api_endpoints, sample_texts):
        """Test endpoints with empty data parameter."""
        # Mock all the models to handle empty data gracefully
        with patch('app.nb_senti_classifier') as mock_nb, \
             patch('app.flair_pre_model') as mock_flair, \
             patch('app.stan_pipeline') as mock_stanza, \
             patch('models.nltk_sentiment_nb.word_tokenize') as mock_tokenize, \
             patch('models.nltk_sentiment_nb.remove_noise') as mock_remove_noise, \
             patch('models.nltk_vader.get_sentiment_vader') as mock_vader, \
             patch('models.textblob_sentiment.get_sentiment_details') as mock_textblob:

            # Configure mocks
            mock_nb.classify.return_value = "Neutral"
            mock_tokenize.return_value = []
            mock_remove_noise.return_value = []
            mock_vader.return_value = {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
            mock_textblob.return_value = {'Polarity_score': 0.0, 'Overall sentiment': 'Neutral'}

            # Mock flair model with proper return values
            mock_flair.predict = Mock()

            # Mock stanza pipeline
            mock_stanza.return_value = Mock()

            # Mock get_unimeasure_sentiment for VADER
            with patch('models.nltk_vader.get_unimeasure_sentiment') as mock_unimeasure:
                mock_unimeasure.return_value = "Neutral"

            for endpoint in api_endpoints:
                for empty_text in sample_texts['empty']:
                    response = client.post(endpoint, json={'data': empty_text})
                    # Empty data should return 400 Bad Request due to input validation
                    assert response.status_code == 400
    
    def test_invalid_json(self, client, api_endpoints):
        """Test endpoints with invalid JSON."""
        for endpoint in api_endpoints:
            response = client.post(endpoint, 
                                 data="invalid json",
                                 content_type='application/json')
            assert response.status_code == 400
    
    def test_get_method_not_allowed(self, client, api_endpoints):
        """Test that GET method is not allowed on sentiment endpoints."""
        for endpoint in api_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 405  # Method Not Allowed
    
    @patch('models.nltk_sentiment_nb.remove_noise')
    @patch('models.nltk_sentiment_nb.word_tokenize')
    @patch('app.nb_senti_classifier')
    def test_special_characters_handling(self, mock_classifier, mock_tokenize, mock_remove_noise, client, sample_texts):
        """Test handling of special characters and URLs."""
        mock_classifier.classify.return_value = "Positive"
        mock_tokenize.return_value = ['test', 'tokens']
        mock_remove_noise.return_value = ['clean', 'tokens']

        for text in sample_texts['special']:
            response = client.post('/sentiment/nb',
                                 json={'data': text})
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data['tweet'] == text
    
    def test_large_text_input(self, client):
        """Test handling of large text input."""
        large_text = "This is a test sentence. " * 1000  # Very long text

        with patch('app.nb_senti_classifier') as mock_classifier, \
             patch('models.nltk_sentiment_nb.word_tokenize') as mock_tokenize, \
             patch('models.nltk_sentiment_nb.remove_noise') as mock_remove_noise:

            mock_classifier.classify.return_value = "Positive"
            mock_tokenize.return_value = ['test', 'tokens']
            mock_remove_noise.return_value = ['clean', 'tokens']

            response = client.post('/sentiment/nb',
                                 json={'data': large_text})
            assert response.status_code == 200
    
    def test_unicode_text_input(self, client):
        """Test handling of Unicode text."""
        unicode_texts = [
            "I love this! 😊❤️🎉",
            "Café résumé naïve",
            "こんにちは世界",  # Japanese
            "Здравствуй мир",  # Russian
            "مرحبا بالعالم"     # Arabic
        ]

        with patch('app.nb_senti_classifier') as mock_classifier, \
             patch('models.nltk_sentiment_nb.word_tokenize') as mock_tokenize, \
             patch('models.nltk_sentiment_nb.remove_noise') as mock_remove_noise:

            mock_classifier.classify.return_value = "Positive"
            mock_tokenize.return_value = ['test', 'tokens']
            mock_remove_noise.return_value = ['clean', 'tokens']

            for text in unicode_texts:
                response = client.post('/sentiment/nb',
                                     json={'data': text})
                assert response.status_code == 200

                data = json.loads(response.data)
                assert data['tweet'] == text


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_404_error(self, client):
        """Test 404 error for non-existent endpoints."""
        response = client.get('/nonexistent')
        assert response.status_code == 404
    
    def test_500_error_simulation(self, client):
        """Test 500 error handling."""
        with patch('app.nb_senti_classifier') as mock_classifier:
            mock_classifier.classify.side_effect = Exception("Model error")
            
            response = client.post('/sentiment/nb', 
                                 json={'data': 'test text'})
            assert response.status_code == 500
            
            data = json.loads(response.data)
            assert 'message' in data  # The actual error response format
            assert data['message'] == 'Sentiment analysis failed'


class TestResponseFormat:
    """Test response format consistency."""
    
    @patch('app.nb_senti_classifier')
    def test_response_structure(self, mock_classifier, client):
        """Test that responses have consistent structure."""
        mock_classifier.classify.return_value = "Positive"
        
        response = client.post('/sentiment/nb', 
                             json={'data': 'test text'})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        
        # Check required fields
        required_fields = ['tweet', 'sentiment', 'model']
        for field in required_fields:
            assert field in data
        
        # Check data types
        assert isinstance(data['tweet'], str)
        assert isinstance(data['sentiment'], str)
        assert isinstance(data['model'], str)
    
    def test_json_content_type(self, client):
        """Test that responses have correct content type."""
        response = client.get('/health')
        assert response.content_type == 'application/json'
