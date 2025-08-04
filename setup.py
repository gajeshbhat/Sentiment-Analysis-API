#!/usr/bin/env python3
"""
Setup script for Sentiment Analysis API

This setup.py is maintained for backward compatibility.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    """Read README.md file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Handle version specifiers
                    if '>=' in line or '==' in line or '<' in line:
                        requirements.append(line)
                    else:
                        requirements.append(line)
    return requirements

setup(
    name="sentiment-analysis-api",
    version="2.0.0",
    author="Gajesh Bhat",
    description="A comprehensive Flask-based REST API for sentiment analysis using multiple NLP libraries",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/gajeshbhat/Sentiment-Analysis-API",
    project_urls={
        "Bug Tracker": "https://github.com/gajeshbhat/Sentiment-Analysis-API/issues",
        "Documentation": "https://github.com/gajeshbhat/Sentiment-Analysis-API#readme",
        "Source Code": "https://github.com/gajeshbhat/Sentiment-Analysis-API",
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'models': ['*.pickle'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentiment-api=app:main",
        ],
    },
    keywords=[
        "sentiment-analysis",
        "nlp",
        "flask",
        "api",
        "machine-learning",
        "nltk",
        "textblob",
        "flair",
        "stanza",
    ],
    zip_safe=False,
)
