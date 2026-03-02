# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

from setuptools import setup, find_packages

setup(
    name="epstein-accountability-index",
    version="0.1.0",
    description="The Accountability Gap - NLP analysis of Epstein case files",
    author="Lindsay Gross",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "spacy>=3.7.2",
        "nltk>=3.8.1",
        "vaderSentiment>=3.3.2",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
        "wikipedia-api>=0.6.0",
        "gdown>=4.7.1",
        "lxml>=4.9.3",
        "plotly>=5.17.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.1",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
        "rapidfuzz>=3.5.2",
    ],
    entry_points={
        "console_scripts": [
            "accountability-index=main:main",
        ],
    },
)
