from setuptools import setup, find_packages
setup(
    name='MLops_DVC_Project',
    version='0.0.0',
    author='Naveen Aswal',
    description='A Machine Learning project with DVC integration',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NaveenAswal/MLops_DVC_Project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.10.18',
    include_package_data=True,
    author_email='gbpec.aswal@gmail.com',
    packages=find_packages(), # Automatically find packages in the directory and search for __init__.py file and consider it as local package
    install_requires=[],  # List your project dependencies here,already have requirements.txt
)

# Data Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import os, re, gc, json, string, pickle
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.sequence import pad_sequences
import pickle
import seaborn as sns
import plotly.express as px
import io
import cufflinks as cf
cf.go_offline()

# NLP
import nltk # type: ignore
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.corpus import stopwords # type: ignore
from nltk import word_tokenize, FreqDist
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from wordcloud import WordCloud
nltk.download('stopwords')
tqdm.pandas()



# Vectorizers and models
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# For interpretability
import shap

# Topic modelling (LDA and BERTopic)
from sklearn.decomposition import LatentDirichletAllocation as LDA
from bertopic import BERTopic

# Sentence embeddings
from sentence_transformers import SentenceTransformer

# spacy model
nlp = spacy.load("en_core_web_sm", disable=['parser'])  # enable tagger/ner/lemmatizer as needed

# Utilities
analyzer = SentimentIntensityAnalyzer()
sns.set(style="whitegrid")
