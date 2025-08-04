# Sentiment Analysis API

A comprehensive Flask-based REST API for sentiment analysis using multiple state-of-the-art NLP libraries and models. This API provides sentiment analysis capabilities through various approaches including NLTK, TextBlob, Flair, and Stanford CoreNLP.

## 🚀 Features

- **Multiple Sentiment Analysis Models**: 
  - NLTK Naive Bayes Classifier (99% accuracy on X (Formerly Twitter) data)
  - NLTK VADER Sentiment Analyzer
  - TextBlob Sentiment Analysis with language detection
  - Flair Pre-trained Sentiment Model
  - Stanford CoreNLP Sentiment Pipeline
  - Combined analysis using multiple models

- **RESTful API**: Clean and intuitive REST endpoints
- **Multi-language Support**: Automatic language detection and translation (TextBlob)
- **Preprocessing Pipeline**: Advanced text cleaning and preprocessing
- **JSON Response Format**: Structured and consistent API responses

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Model Details](#model-details)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/gajeshbhat/Sentiment-Analysis-API.git
   cd Sentiment-Analysis-API
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models** (automatically handled on first run)
   - NLTK data packages
   - Stanza English model
   - Flair pre-trained sentiment model

## 🚀 Quick Start

1. **Start the API server**
   ```bash
   python app.py
   ```

2. **Test the API**
   ```bash
   curl -X POST http://localhost:5000/sentiment/vader \
        -H "Content-Type: application/json" \
        -d '{"data": "I love this amazing product!"}'
   ```

## 📡 API Endpoints

### Base URL: `http://localhost:5000`

| Endpoint | Method | Description | Model Used |
|----------|--------|-------------|------------|
| `/sentiment/nb` | POST | NLTK Naive Bayes | Pre-trained NB classifier |
| `/sentiment/vader` | POST | NLTK VADER | VADER lexicon |
| `/sentiment/tb` | POST | TextBlob | TextBlob sentiment |
| `/sentiment/fl` | POST | Flair | Flair pre-trained model |
| `/sentiment/scnlp` | POST | Stanford CoreNLP | Stanza pipeline |
| `/sentiment/all` | POST | Combined Analysis | Stanford + Flair |

### Request Format

All endpoints accept POST requests with JSON payload:

```json
{
  "data": "Your text to analyze here"
}
```

### Response Formats

#### NLTK Naive Bayes (`/sentiment/nb`)
```json
{
  "tweet": "I love this product!",
  "sentiment": "Positive"
}
```

#### NLTK VADER (`/sentiment/vader`)
```json
{
  "tweet": "I love this product!",
  "sentiment_score": {
    "overall_sentiment": "Positive",
    "polarity_scores": {
      "neg": 0.0,
      "neu": 0.323,
      "pos": 0.677,
      "compound": 0.6369
    }
  }
}
```

#### TextBlob (`/sentiment/tb`)
```json
{
  "tweet": "I love this product!",
  "sentiment_scores": {
    "Polarity_score": 0.5,
    "subjectivity_score": 0.6,
    "assessment_details": [["love", 0.5, 0.6, null]],
    "Overall sentiment": "Positive"
  }
}
```

#### Flair (`/sentiment/fl`)
```json
{
  "tweet": "I love this product!",
  "sentiment_scores": {
    "overall_sentiment": "POSITIVE",
    "polarity_score": "0.9998"
  }
}
```

#### Stanford CoreNLP (`/sentiment/scnlp`)
```json
{
  "tweet": "I love this product!",
  "sentiment_scores": 2
}
```

#### Combined Analysis (`/sentiment/all`)
```json
{
  "tweet": "I love this product!",
  "sentiment_scores": {
    "stanford_core_nlp": 2,
    "flair": {
      "overall_sentiment": "POSITIVE",
      "polarity_score": "0.9998"
    }
  }
}
```

## 💡 Usage Examples

### Python Example

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/sentiment/vader"

# Sample text
data = {
    "data": "This movie is absolutely fantastic! I loved every moment of it."
}

# Make request
response = requests.post(url, json=data)
result = response.json()

print(json.dumps(result, indent=2))
```

### cURL Example

```bash
# Analyze sentiment using VADER
curl -X POST http://localhost:5000/sentiment/vader \
     -H "Content-Type: application/json" \
     -d '{"data": "This is an amazing day!"}'

# Analyze using multiple models
curl -X POST http://localhost:5000/sentiment/all \
     -H "Content-Type: application/json" \
     -d '{"data": "I hate waiting in long queues."}'
```

### JavaScript Example

```javascript
const analyzeSentiment = async (text) => {
  const response = await fetch('http://localhost:5000/sentiment/vader', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ data: text })
  });
  
  const result = await response.json();
  console.log(result);
};

analyzeSentiment("I'm so excited about this new project!");
```

## 🧠 Model Details

### NLTK Naive Bayes
- **Accuracy**: ~99% on tweet data
- **Training**: Pre-trained on tweet sentiment dataset
- **Output**: Binary classification (Positive/Negative)
- **Best for**: Social media text, informal language

### NLTK VADER
- **Type**: Lexicon-based approach
- **Output**: Compound score (-1 to 1) + individual scores
- **Best for**: Social media text, handles emoticons and slang
- **Threshold**: Positive (≥0.05), Neutral (-0.05 to 0.05), Negative (<-0.05)

### TextBlob
- **Features**: Polarity and subjectivity scores
- **Language Support**: Auto-detection and translation
- **Output**: Polarity (-1 to 1), Subjectivity (0 to 1)
- **Best for**: General text, multi-language support

### Flair
- **Type**: Deep learning-based
- **Model**: Pre-trained transformer model
- **Output**: Confidence score with sentiment label
- **Best for**: High accuracy on diverse text types

### Stanford CoreNLP (Stanza)
- **Type**: Neural network-based
- **Output**: Integer sentiment score (0-4 scale)
- **Best for**: Formal text, news articles

## 🔧 Development

### Project Structure

```
Sentiment-Analysis-API/
├── app.py                 # Main Flask application
├── models/               # Sentiment analysis models
│   ├── nltk_sentiment_nb.py
│   ├── nltk_vader.py
│   ├── textblob_sentiment.py
│   └── tweet_senti_nb.pickle
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md           # This file
```

### Adding New Models

1. Create a new model file in the `models/` directory
2. Implement the sentiment analysis logic
3. Create a new Resource class in `app.py`
4. Add the endpoint using `api.add_resource()`

### Environment Variables

You can configure the application using environment variables:

- `FLASK_ENV`: Set to `development` for debug mode
- `FLASK_PORT`: Port number (default: 5000)

## 🐳 Docker Deployment

### Quick Start with Docker

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the API**
   - API: http://localhost:5000
   - Health check: http://localhost:5000/health

### Manual Docker Build

```bash
# Build the image
docker build -t sentiment-analysis-api .

# Run the container
docker run -p 5000:5000 sentiment-analysis-api
```

### Docker Features

- **Health checks**: Automatic container health monitoring
- **Volume mounts**: Persistent logs and model storage
- **Production ready**: Optimized for deployment
- **Easy scaling**: Ready for orchestration platforms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NLTK team for natural language processing tools
- Flair team for the excellent NLP framework
- Stanford NLP Group for CoreNLP
- TextBlob developers for the simple API

## 📞 Support

If you have any questions or issues, please:

1. Check the [Issues](https://github.com/gajeshbhat/Sentiment-Analysis-API/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about your environment and the issue

---

**Made with ❤️ by [Gajesh Bhat](https://www.gajeshbhat.com)**
