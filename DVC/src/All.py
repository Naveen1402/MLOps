import os

import textstat
import setup
from nltk import word_tokenize, sent_tokenize
from setup import analyzer
from sklearn.decomposition import LatentDirichletAllocation as LDA

"""
Python's built-in glob module provides similar functionality.
enabling you to easily find files and directories matching specific patterns
"""


# Data Loading

def load_data(file_path: str) -> pd.DataFrame:
    """
    Universal Data Loading Function
    Supports: CSV, Excel, JSON, Parquet, TXT (delimited)
    Includes error handling and informative messages.
    """
    try:
        # 1. Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        # 2. Identify file extension
        ext = os.path.splitext(file_path)[1].lower()

        # 3. Load based on extension
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif ext == '.txt':
            df = pd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError(f"⚠️ Unsupported file type: {ext}")

        # 4. Success message
        print(f"✅ Successfully loaded {file_path}")
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        return df

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except pd.errors.EmptyDataError:
        print("⚠️ The file is empty.")
    except pd.errors.ParserError:
        print("⚠️ Parsing error: check file delimiter or encoding.")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")

#Data Merging
import re
import pandas as pd

def merge_ted_transcripts(ted: pd.DataFrame, trans: pd.DataFrame) -> pd.DataFrame:
    """
    Merge TED metadata (ted) and transcripts (trans) using the best available key.

    Priority of merge keys: 'url', 'slug', 'title', 'talk_id', 'id'.
    If none of these exist in both, falls back to merging on a cleaned title.

    Parameters
    ----------
    ted : pd.DataFrame
        TED talks metadata.
    trans : pd.DataFrame
        TED talk transcripts.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    merge_keys = None
    for k in ['url', 'slug', 'title', 'talk_id', 'id']:
        if k in ted.columns and k in trans.columns:
            merge_keys = k
            break

    if merge_keys:
        df = ted.merge(trans, how='left', on=merge_keys, suffixes=('', '_trans'))
        print(f"Merged on '{merge_keys}' — shape: {df.shape}")
    else:
        # fallback: clean titles for fuzzy matching
        def clean_title(s):
            return re.sub(r'\W+', ' ', str(s).lower()).strip()

        ted['title_cl'] = ted.get('title', ted.get('name', '')).apply(clean_title)
        trans['title_cl'] = trans.get('title', trans.get('name', '')).apply(clean_title)
        df = ted.merge(trans, how='left', on='title_cl', suffixes=('', '_trans'))
        print(f"Merged on cleaned title — shape: {df.shape}")

    return df

# Data PreProcessing
import pandas as pd

def Basic_preprocess_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Perform minimal preprocessing and sanity checks before EDA or MLOps pipelines.

    Steps:
    1. Basic info + NA count summary
    2. Convert date columns to datetime (if present)
    3. Convert numeric columns (views, likes, comments, etc.)
    4. Normalize text fields (title, description, transcript)
    5. Ensure consistent dtypes and clean text columns

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to clean.
    verbose : bool, optional
        If True, prints basic info and NA summary.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for downstream use.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    # -------------------------------
    # 1. Basic info & sanity check
    # -------------------------------
    if verbose:
        print("\n🔍 DataFrame Info:")
        try:
            df.info()
            print("\n🧩 Missing values (top 30):")
            print(df.isna().sum().sort_values(ascending=False).head(30))
        except Exception as e:
            print(f"⚠️ Could not display basic info: {e}")

    # -------------------------------
    # 2. Date conversion
    # -------------------------------
    date_cols = ['film_date', 'published_date', 'date', 'date_added']
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if verbose:
                    print(f"🕓 Converted '{col}' to datetime.")
            except Exception as e:
                print(f"⚠️ Failed to convert '{col}' to datetime: {e}")

    # -------------------------------
    # 3. Numeric casting
    # -------------------------------
    num_cols = ['views', 'comments', 'likes', 'ratings', 'num_speaker_talks']
    for col in num_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if verbose:
                    print(f"🔢 Converted '{col}' to numeric.")
            except Exception as e:
                print(f"⚠️ Failed to convert '{col}' to numeric: {e}")

    # -------------------------------
    # 4. Text field normalization
    # -------------------------------
    try:
        # Try to find transcript column if missing
        if 'transcript' not in df.columns:
            for alt in ['transcripts', 'text', 'content', 'body']:
                if alt in df.columns:
                    df.rename(columns={alt: 'transcript'}, inplace=True)
                    if verbose:
                        print(f"🗒️ Renamed '{alt}' to 'transcript'.")
                    break

        df['transcript'] = df.get('transcript', '').fillna('').astype(str)
        df['description'] = df.get('description', '').fillna('').astype(str)
        df['title'] = df.get('title', df.get('name', '')).fillna('').astype(str)
    except Exception as e:
        print(f"⚠️ Text field setup failed: {e}")

    # -------------------------------
    # 5. Basic text cleanup
    # -------------------------------
    text_cols = ['title', 'description', 'transcript']
    for col in text_cols:
        if col in df.columns:
            try:
                df[col] = df[col].str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            except Exception as e:
                print(f"⚠️ Could not clean text column '{col}': {e}")

    if verbose:
        print(f"\n✅ Preprocessing complete. Final shape: {df.shape}")

    return df


# Advanced EDA

def automated_eda(df, text_col=None, log_numeric_cols=None, n_topics=10, max_features=20000):
    """
    Performs automated EDA for MLOps pipelines.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_col : str
        Name of the text column for text-based features.
    log_numeric_cols : list of str
        Columns to log-transform automatically (e.g., 'views').
    n_topics : int
        Number of topics for LDA.
    max_features : int
        Max features for TF-IDF vectorizer.
    
    Returns
    -------
    pd.DataFrame
        Original dataframe enriched with automated numeric and text features.
    dict
        Summary information: missing percentages, skewness, topic keywords.
    """
    
    summary = {}
    df = df.copy()
    
    # 1️⃣ Numeric missingness
    missing_pct = (df.isna().sum() / len(df)).sort_values(ascending=False)
    summary['missing_pct'] = missing_pct
    
    # 2️⃣ Numeric log-transform if specified
    if log_numeric_cols:
        for col in log_numeric_cols:
            if col in df.columns:
                df[col + "_log"] = np.log1p(df[col])
    
    # 3️⃣ Text-based features
    if text_col and text_col in df.columns:
        def text_stats(t):
            t = str(t)
            words = word_tokenize(t)
            sents = sent_tokenize(t)
            wc = len(words)
            uc = len(set([w.lower() for w in words]))
            return pd.Series({
                'len_chars': len(t),
                'len_words': wc,
                'len_sents': len(sents),
                'unique_word_ratio': uc / max(wc, 1),
                'avg_word_len': np.mean([len(w) for w in words]) if wc > 0 else 0
            })
        
        text_features = df[text_col].apply(text_stats)
        df = pd.concat([df, text_features], axis=1)
        
        # 4️⃣ Sentiment and readability
        df['vader_compound'] = df[text_col].apply(lambda t: analyzer.polarity_scores(str(t))['compound'])
        df['flesch_reading'] = df[text_col].apply(
            lambda t: textstat.flesch_reading_ease(str(t)) if len(str(t)) > 20 else np.nan
        )
        
        # 5️⃣ LDA topic modeling (optional, lightweight)
        tf = TfidfVectorizer(max_features=max_features, stop_words='english')
        Xtf = tf.fit_transform(df[text_col].fillna(''))
        lda = LDA(n_components=n_topics, random_state=42)
        lda.fit(Xtf)
        
        feature_names = tf.get_feature_names_out()
        topic_keywords = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-11:-1]  # top 10 words
            topic_keywords[topic_idx] = [feature_names[i] for i in top_features_ind]
        
        summary['topic_keywords'] = topic_keywords
    
    return df, summary

"""
df_processed, eda_summary = automated_eda(
    df, 
    text_col='transcript', 
    log_numeric_cols=['views'], 
    n_topics=5
)

print(eda_summary['missing_pct'].head())
print(eda_summary['topic_keywords'])

"""
# Final TextPreprocessing

import re
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Downloads
nltk.download('stopwords')
tqdm.pandas()

# Regex patterns
url_re = re.compile(r'https?://\S+|www\.\S+')
brackets = re.compile(r'\[.*?\]|\(.*?\)')
multi_space = re.compile(r'\s+')
repeat = re.compile(r'(.)\1{2,}')  # char repeated 3+ times

# Stopwords
stop_words = set(stopwords.words('english'))
domain_stop = set(['applause','laughs','laugh','audience'])
stop_words |= domain_stop

# SpaCy model
nlp = spacy.load("en_core_web_sm", disable=['parser','ner'])

# Sentence embedding model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


def preprocess_text_pipeline(df, text_col, ngram_range=(1,2), max_ngram_features=2000, tfidf_max_features=5000):
    """
    Full text preprocessing for MLOps pipelines.
    Steps:
        1. Clean text
        2. Lemmatize
        3. POS and NER counts
        4. Extract n-gram counts
        5. TF-IDF features
        6. Sentence embeddings
    
    Returns:
        df with additional feature columns
        dict containing vectorizers, embeddings, metadata
    """
    meta = {}
    
    def safe_apply(func, x):
        try:
            return func(x)
        except Exception as e:
            print(f"Error processing row: {e}")
            return np.nan

    # 1️⃣ Clean text
    def clean_text(text):
        text = str(text)
        text = url_re.sub(' ', text)
        text = brackets.sub(' ', text)
        text = repeat.sub(r'\1', text)
        text = re.sub(r'[^A-Za-z0-9\'\s\-]', ' ', text)
        text = multi_space.sub(' ', text)
        return text.strip().lower()
    
    df[text_col + '_clean'] = df[text_col].progress_map(lambda x: safe_apply(clean_text, x))
    
    # 2️⃣ Lemmatization
    def lemmatize_text(doc):
        doc = nlp(str(doc))
        tokens = []
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            lemma = tok.lemma_.lower()
            if lemma in stop_words or lemma.isnumeric() or lemma == '-pron-':
                continue
            tokens.append(lemma)
        return " ".join(tokens)
    
    df[text_col + '_lemma'] = df[text_col + '_clean'].progress_map(lambda x: safe_apply(lemmatize_text, x))
    
    # 3️⃣ POS counts & NER
    def pos_ner_counts(text):
        doc = nlp(str(text))
        pos_counts = {}
        for p in ['NOUN','VERB','ADJ','ADV','PROPN']:
            pos_counts[f'pos_{p.lower()}'] = sum(1 for t in doc if t.pos_==p)
        pos_counts['num_ents'] = len(doc.ents)
        return pd.Series(pos_counts)
    
    pos_feats = df[text_col + '_clean'].progress_map(lambda x: safe_apply(pos_ner_counts, x))
    df = pd.concat([df, pos_feats], axis=1)
    
    # 4️⃣ N-gram counts
    cv = CountVectorizer(ngram_range=ngram_range, max_features=max_ngram_features, stop_words='english')
    try:
        X_cv = cv.fit_transform(df[text_col + '_lemma'].fillna(''))
        ngram_cols = [f"ngram_{w}" for w in cv.get_feature_names_out()]
        df_ngrams = pd.DataFrame(X_cv.toarray(), columns=ngram_cols, index=df.index)
        df = pd.concat([df, df_ngrams], axis=1)
        meta['ngram_vectorizer'] = cv
    except Exception as e:
        print(f"N-gram extraction failed: {e}")
    
    # 5️⃣ TF-IDF
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english')
    try:
        X_tfidf = tfidf.fit_transform(df[text_col + '_lemma'].fillna(''))
        tfidf_cols = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
        df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_cols, index=df.index)
        df = pd.concat([df, df_tfidf], axis=1)
        meta['tfidf_vectorizer'] = tfidf
    except Exception as e:
        print(f"TF-IDF extraction failed: {e}")
    
    # 6️⃣ Sentence embeddings
    try:
        embeddings = sbert_model.encode(df[text_col + '_lemma'].fillna('').tolist(), show_progress_bar=True)
        embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
        df_embed = pd.DataFrame(embeddings, columns=embed_cols, index=df.index)
        df = pd.concat([df, df_embed], axis=1)
        meta['sentence_embeddings'] = embeddings
    except Exception as e:
        print(f"Sentence embeddings failed: {e}")
    
    return df, meta


'''
df_processed, eda_meta = preprocess_text_pipeline(df, text_col='transcript')

print(df_processed.head())
print(eda_meta.keys())  # vectorizers, embeddings

'''

# Feature Engineering(Text metadata)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.preprocessing import StandardScaler
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")


def build_features_MLOps(
    df,
    text_col='transcript',
    lemma_col='transcript_lemma',
    title_col='title',
    speaker_col='speaker',
    metadata_cols=['film_year', 'duration', 'language', 'event'],
    tfidf_max_features=30000,
    tfidf_ngram_range=(1,2),
    min_df=5,
    use_topic_model=True,
    verbose=True
):
    """
    Generic, fault-tolerant feature engineering function for MLOps pipelines.
    
    Combines:
      - Metadata (film_year, duration, etc.)
      - Speaker experience
      - Text features (TF-IDF, Sentence Embeddings, Topic Probabilities)
      - Derived features (sentiment, readability, POS counts, interactions)
    
    Returns:
      features_df (pd.DataFrame)
      components (dict): trained models & vectorizers
    """
    features_df = pd.DataFrame(index=df.index)
    components = {}
    
    # ==============================
    # 1️⃣ METADATA + SPEAKER FEATURES
    # ==============================
    try:
        for col in metadata_cols:
            if col in df.columns:
                features_df[col] = df[col]
        if speaker_col in df.columns and title_col in df.columns:
            df['speaker_talk_count'] = df.groupby(speaker_col)[title_col].transform('count')
            features_df['speaker_talk_count'] = df['speaker_talk_count']
        if verbose: print("✅ Metadata features added.")
    except Exception as e:
        print(f"⚠️ Metadata feature creation failed: {e}")
    
    # ==============================
    # 2️⃣ READABILITY + SENTIMENT
    # ==============================
    try:
        analyzer = SentimentIntensityAnalyzer()
        df['flesch_reading'] = df[text_col].apply(lambda x: flesch_reading_ease(str(x)) if isinstance(x, str) else np.nan)
        df['vader_compound'] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))['compound'] if isinstance(x, str) else np.nan)
        df['len_words'] = df[text_col].apply(lambda x: len(str(x).split()))
        df['unique_word_ratio'] = df[text_col].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-6))
        if verbose: print("✅ Readability & sentiment features added.")
    except Exception as e:
        print(f"⚠️ Readability/Sentiment failed: {e}")
    
    # ==============================
    # 3️⃣ INTERACTION FEATURES
    # ==============================
    try:
        df['interaction_len_flesch'] = df['len_words'] * df['flesch_reading']
        df['interaction_vader_uniq'] = df['vader_compound'] * df['unique_word_ratio']
        features_df[['flesch_reading', 'vader_compound', 'len_words', 'unique_word_ratio',
                     'interaction_len_flesch', 'interaction_vader_uniq']] = df[
            ['flesch_reading', 'vader_compound', 'len_words', 'unique_word_ratio',
             'interaction_len_flesch', 'interaction_vader_uniq']
        ]
        if verbose: print("✅ Interaction features computed.")
    except Exception as e:
        print(f"⚠️ Interaction feature creation failed: {e}")
    
    # ==============================
    # 4️⃣ TF-IDF VECTORS
    # ==============================
    try:
        tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range, min_df=min_df, stop_words='english')
        X_tfidf = tfidf.fit_transform(df[lemma_col].fillna(''))
        components['tfidf'] = tfidf
        features_df = pd.concat([features_df, pd.DataFrame(X_tfidf.toarray(), columns=[f"tfidf_{i}" for i in range(X_tfidf.shape[1])], index=df.index)], axis=1)
        if verbose: print(f"✅ TF-IDF extracted: {X_tfidf.shape}")
    except Exception as e:
        print(f"⚠️ TF-IDF extraction failed: {e}")
    
    # ==============================
    # 5️⃣ SENTENCE EMBEDDINGS
    # ==============================
    try:
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = sbert.encode(df[text_col].astype(str).tolist(), show_progress_bar=verbose, batch_size=16)
        emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
        df_embed = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
        features_df = pd.concat([features_df, df_embed], axis=1)
        components['sentence_transformer'] = sbert
        if verbose: print(f"✅ Sentence embeddings computed: {embeddings.shape}")
    except Exception as e:
        print(f"⚠️ Sentence embeddings failed: {e}")
    
    # ==============================
    # 6️⃣ TOPIC MODELING (BERTopic)
    # ==============================
    if use_topic_model:
        try:
            umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
            topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, min_topic_size=20, verbose=verbose)
            
            topics, probs = topic_model.fit_transform(df[text_col].astype(str).tolist())
            df['topic'] = topics
            df['topic_prob'] = [max(p) if isinstance(p, (list, np.ndarray)) else np.nan for p in probs]
            
            topic_dummies = pd.get_dummies(df['topic'], prefix='topic', dummy_na=True)
            features_df = pd.concat([features_df, topic_dummies, df[['topic_prob']]], axis=1)
            
            components['topic_model'] = topic_model
            if verbose: print(f"✅ Topics modeled: {len(set(topics))} topics.")
        except Exception as e:
            print(f"⚠️ Topic modeling failed: {e}")
    
    # ==============================
    # 7️⃣ FINAL SCALING
    # ==============================
    try:
        scaler = StandardScaler()
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols].fillna(0))
        components['scaler'] = scaler
        if verbose: print("✅ Features scaled and ready for modeling.")
    except Exception as e:
        print(f"⚠️ Feature scaling failed: {e}")
    
    return features_df, components


"""
features_df, components = build_features_MLOps(df, text_col='transcript', lemma_col='transcript_lemma')

print(features_df.shape)
print(features_df.columns[:20])

"""

import numpy as np
import pandas as pd

def prepare_targets_MLOps(
    df,
    target_col='views',
    classification_quantile=0.9,
    log_transform=True,
    verbose=True
):
    """
    Generic, fault-tolerant target creation function for MLOps pipelines.

    Creates:
      - Regression target (log-transformed if specified)
      - Binary classification target based on quantile threshold (e.g., top 10%)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing target column.
    target_col : str
        Name of the column containing the raw target (e.g., 'views').
    classification_quantile : float
        Quantile threshold for binary classification (default = 0.9 for top 10% popular).
    log_transform : bool
        Whether to create log-transformed regression target.
    verbose : bool
        If True, prints process status and stats.

    Returns
    -------
    df : pd.DataFrame
        Updated DataFrame with new target columns:
          - 'target_log' (regression)
          - 'target_class' (binary classification)
    target_info : dict
        Metadata about thresholds and transformations used.
    """
    target_info = {}

    # ===============================
    # 1️⃣ Validate input
    # ===============================
    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in DataFrame.")
    if df[target_col].isnull().all():
        raise ValueError(f"❌ Target column '{target_col}' contains only NaN values.")

    # ===============================
    # 2️⃣ Regression target (log-transform optional)
    # ===============================
    try:
        if log_transform:
            df['target_log'] = np.log1p(df[target_col].fillna(0))
            target_info['regression_target'] = 'log1p-transformed'
        else:
            df['target_log'] = df[target_col].fillna(0)
            target_info['regression_target'] = 'raw'
        if verbose:
            print(f"✅ Regression target ('target_log') prepared using log1p={log_transform}")
    except Exception as e:
        print(f"⚠️ Failed to create regression target: {e}")
        df['target_log'] = np.nan

    # ===============================
    # 3️⃣ Classification target (top quantile)
    # ===============================
    try:
        threshold = df[target_col].quantile(classification_quantile)
        df['target_class'] = (df[target_col] >= threshold).astype(int)
        target_info['classification_threshold'] = float(threshold)
        target_info['classification_quantile'] = classification_quantile
        if verbose:
            pos_rate = df['target_class'].mean()
            print(f"✅ Classification target ('target_class') created at quantile={classification_quantile:.2f}")
            print(f"   → Threshold: {threshold:,.0f} | Positive rate: {pos_rate*100:.2f}%")
    except Exception as e:
        print(f"⚠️ Failed to create classification target: {e}")
        df['target_class'] = np.nan
        target_info['classification_threshold'] = None

    # ===============================
    # 4️⃣ Return results
    # ===============================
    return df, target_info


"""
df, target_info = prepare_targets_MLOps(df, target_col='views', classification_quantile=0.9)

print(df[['views', 'target_log', 'target_class']].head())
print(target_info)

"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
from scipy.sparse import hstack
from scipy import sparse

def train_model_MLOps(
    train_df,
    test_df,
    task='regression',  # 'regression' or 'classification'
    features=None,      # embeddings or TF-IDF matrix
    meta_cols=None,     # list of numeric metadata columns
    target_col=None,    # regression: 'log_views', classification: 'popular'
    tfidf=None,         # optional TF-IDF vectorizer (for classification)
    verbose=True
):
    """
    Generic model training function with error handling.
    Supports:
      - Regression: LightGBM with embeddings + meta
      - Classification: XGBoost with TF-IDF + meta
    
    Returns:
      model: trained model object
      metrics: dict with performance metrics
    """
    metrics = {}
    model = None

    try:
        if task == 'regression':
            if features is None or meta_cols is None:
                raise ValueError("Embeddings and meta_cols must be provided for regression.")
            
            # Stack embeddings + meta numeric
            X_train = np.hstack([np.array(features)[train_df.index], train_df[meta_cols].fillna(0).values])
            X_test  = np.hstack([np.array(features)[test_df.index],  test_df[meta_cols].fillna(0).values])
            y_train = train_df[target_col].values
            y_test  = test_df[target_col].values

            lgb_train = lgb.Dataset(X_train, label=y_train)
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_freq': 5,
                'bagging_fraction': 0.8,
                'seed': 42
            }
            model = lgb.train(params, lgb_train, num_boost_round=1000,
                              valid_sets=[lgb_train],
                              callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
                              verbose_eval=verbose)
            
            preds = model.predict(X_test, num_iteration=model.best_iteration)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, preds))
            metrics['r2']   = r2_score(y_test, preds)
            if verbose:
                print(f"✅ Regression done. RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

        elif task == 'classification':
            if tfidf is None or meta_cols is None:
                raise ValueError("TF-IDF vectorizer and meta_cols must be provided for classification.")

            # Transform TF-IDF + stack with meta
            X_tfidf_train = tfidf.transform(train_df['transcript_lemma'])
            X_tfidf_test  = tfidf.transform(test_df['transcript_lemma'])
            meta_train = train_df[meta_cols].fillna(0).values
            meta_test  = test_df[meta_cols].fillna(0).values

            X_train = hstack([X_tfidf_train, sparse.csr_matrix(meta_train)])
            X_test  = hstack([X_tfidf_test,  sparse.csr_matrix(meta_test)])
            y_train = train_df[target_col].values
            y_test  = test_df[target_col].values

            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='auc',
                use_label_encoder=False,
                random_state=42
            )

            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=verbose)

            yprob = model.predict_proba(X_test)[:,1]
            metrics['auc'] = roc_auc_score(y_test, yprob)
            metrics['report'] = classification_report(y_test, model.predict(X_test))
            if verbose:
                print(f"✅ Classification done. AUC: {metrics['auc']:.4f}")
                print(metrics['report'])
        else:
            raise ValueError(f"Task must be 'regression' or 'classification'. Got {task}")

    except Exception as e:
        print(f"❌ Model training failed: {e}")
    
    return model, metrics

"""
Regression (LightGBM)

model_lgb, metrics_lgb = train_model_MLOps(
    train_df=train_df,
    test_df=test_df,
    task='regression',
    features=embeddings,
    meta_cols=meta_cols,
    target_col='log_views'
)


Classification (XGBoost)

model_xgb, metrics_xgb = train_model_MLOps(
    train_df=train_df,
    test_df=test_df,
    task='classification',
    tfidf=tfidf,
    meta_cols=meta_cols,
    target_col='popular'
)
"""

# MOdel Tuning

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def tune_model_MLOps(
    model,
    param_grid,
    X,
    y,
    task='classification',  # 'classification' or 'regression'
    cv_splits=5,
    scoring=None,
    n_jobs=-1,
    verbose=2
):
    """
    Generic function for model hyperparameter tuning with GridSearchCV.
    
    Parameters
    ----------
    model : sklearn estimator
        Classifier or regressor instance
    param_grid : dict
        Hyperparameters grid to search
    X : array-like or sparse
        Feature matrix
    y : array-like
        Target vector
    task : str
        'classification' or 'regression'
    cv_splits : int
        Number of cross-validation folds
    scoring : str or callable
        Scoring metric. If None, defaults to 'roc_auc' for classification, 'neg_root_mean_squared_error' for regression
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level for GridSearchCV

    Returns
    -------
    best_model : fitted estimator
        Best estimator found
    best_params : dict
        Best hyperparameters
    best_score : float
        Best cross-validation score
    """
    best_model = None
    best_params = None
    best_score = None
    
    try:
        # Define default scoring
        if scoring is None:
            if task == 'classification':
                scoring = 'roc_auc'
                cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            elif task == 'regression':
                scoring = 'neg_root_mean_squared_error'
                cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
            else:
                raise ValueError("task must be 'classification' or 'regression'")
        else:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42) if task=='classification' else KFold(n_splits=cv_splits, shuffle=True, random_state=42)

        # Initialize GridSearchCV
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Fit
        gs.fit(X, y)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        best_score = gs.best_score_

        print(f"✅ GridSearch completed. Best {scoring}: {best_score:.4f}")
        print("Best parameters:", best_params)
        
    except Exception as e:
        print(f"❌ GridSearch failed: {e}")
    
    return best_model, best_params, best_score

"""
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [6, 12],
    'min_samples_leaf': [1, 3]
}

rfc = RandomForestClassifier(random_state=42, n_jobs=-1)

best_model, best_params, best_score = tune_model_MLOps(
    model=rfc,
    param_grid=param_grid,
    X=X_train_comb,
    y=y_train_cl,
    task='classification'
)

"""
