import pandas as pd
import numpy as np

from datetime import datetime
import sys
import os
username = os.environ.get('USER')
sys.path.append(f'/data/workspace/{username}')

sys.path.append(os.path.join(os.path.abspath("../../"), "src"))
from extract_header_features import *
from extract_text_features import *
from extract_url_features import *
from extract_text_keywords import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
)

from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.compose import make_column_transformer

from sklearn.metrics import confusion_matrix

from scipy.stats import expon, lognorm, loguniform, randint, uniform, norm

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier


# Hide warnings
import warnings
warnings.filterwarnings('ignore')

import pickle

startTime = datetime.now()

original_df = pd.read_parquet('/data/workspace/dataset/full-dataset/raw/train.parquet')
input_df = pd.read_parquet('/data/workspace/dataset/full-dataset/processed/train.parquet')
combined_df = original_df.join(input_df)
df_without_sp = combined_df[combined_df['target_3'] != 'self_phishing'].copy()

print("Data loaded successfully.")

train_df, test_df = train_test_split(df_without_sp, test_size=0.3, random_state=42)

list_cols = ["Content_types", "attachment_types", "urls"]

for col in list_cols:
    train_df[col] = train_df[col].apply(lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else str(x))
    test_df[col] = test_df[col].apply(lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else str(x))


train_df["Subject"] = train_df["Subject"].fillna("")
train_df["text_preprocessed"] = train_df["text_preprocessed"].fillna("")

test_df["Subject"] = test_df["Subject"].fillna("")
test_df["text_preprocessed"] = test_df["text_preprocessed"].fillna("")

X_train = train_df.drop(columns=['target_1'])
y_train = train_df['target_1']

X_test = test_df.drop(columns=['target_1'])
y_test = test_df['target_1']

label_encoder = LabelEncoder()
y_train_num = label_encoder.fit_transform(y_train)
y_test_num = label_encoder.transform(y_test)

print("Train and test sets prepared successfully.")

numeric_transformer = make_pipeline(StandardScaler())

binary_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore', drop='if_binary'))

categorical_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore', drop='if_binary'))


header_numeric_feats = [
    "routing_length"
]

header_binary_feats = [
    "is_multipart",
    "dmarc_authentication_present", 
    "dkim_sender_domains_match",
    "attachments_present", 
    "to_from_addresses_match", 
    "sender_email_spf_match"
]

header_categorical_feats = [
    "From_name", 
    "From_email", 
    "From_email_domain", 
    "To_name", 
    "To_email", 
    "To_email_domain",
    "dkim_result",
    "spf_result", 
    "dmarc_result", 
    "Content-Language"
]

header_text_feats = [
    "Subject"
]

header_drop_feats = [
    "From",                         # Info extracted to From_name, From_email, From_email_domain
    "To",                           # Info extracted to To_name, To_email, To_email_domain
    "Received",                     # Info extracted to routing_length
    "Authentication-Results",       # Info extracted to dmarc_authentication_present, dkim_result, spf_result, dmarc_result
    "received-spf",                 # Info extracted to spf_result, sender_email_spf_match
    "DKIM-Signature",               # Info extracted to dkim_sender_domains_match
    "Reply-To",                     # Mostly missing, not useful
    "Return-Path",                  # Mostly missing, not useful
    "text_plain",                   
    "text_clean", 
    "text_html"
]

subject_vectorizer = make_pipeline(CountVectorizer())

preprocessor_header = make_column_transformer(
    ("passthrough", header_numeric_feats),
    (binary_transformer, header_binary_feats),
    (categorical_transformer, header_categorical_feats),
    (subject_vectorizer, header_text_feats[0]), # Subject
    remainder='drop'
)


body_numeric_feats = [
        "word_count",
        "readable_proportion",
        "whitespace_ratio",
        "alphabet_proportion",
        "grammar_error_rate",
        "english_french_proportion",
        "url_count"
]

body_binary_feats = [
        "non_ascii_present",
        "hidden_text_present",
        "all_urls_accessible",
        "urls_redirected",
        "ip_addr_urls",
        "http_urls_present",
        "url_at_symbol",
        "url_port_number",
        "any_long_urls",
        "url_multiple_subdomains"
]

body_categorical_feats = [
        "html_parsing_error"
]

body_text_feats = [
        "Content_types",
        "attachment_types",
        "text_preprocessed",
        "urls"
]

content_types_vectorizer = make_pipeline(CountVectorizer())
attachment_types_vectorizer = make_pipeline(CountVectorizer())
text_preprocessed_vectorizer = make_pipeline(CountVectorizer())
urls_vectorizer = make_pipeline(CountVectorizer())


preprocessor_body = make_column_transformer(
        (numeric_transformer, body_numeric_feats),
        (binary_transformer, body_binary_feats),
        (categorical_transformer, body_categorical_feats),
        (content_types_vectorizer, body_text_feats[0]), # content_types
        (attachment_types_vectorizer, body_text_feats[1]), # attachment_types
        (text_preprocessed_vectorizer, body_text_feats[2]), # text_preprocessed
        (urls_vectorizer, body_text_feats[3]), # urls
        remainder='drop'
)

print("Preprocessors created successfully.")

pipe_header = make_pipeline(
    preprocessor_header,
    XGBClassifier(n_jobs=-1, eval_metric="error", objective="binary:logistic")
)

pipe_body = make_pipeline(
    preprocessor_body,
    XGBClassifier(n_jobs=-1, eval_metric="error", objective="binary:logistic")
)

estimator = [("header", pipe_header), ("body", pipe_body)]

stacking = StackingClassifier(
    estimators=estimator,
    final_estimator=SVC(kernel='rbf', class_weight='balanced', random_state=123),
    n_jobs=-1
)

param_dist_stacking = {
    "header__columntransformer__pipeline-3__countvectorizer__max_features": [5000, 10000, 15000, 20000, 25000, 30000],
    "header__columntransformer__pipeline-3__countvectorizer__max_df": [0.8, 0.9, 1.0],
    "header__columntransformer__pipeline-3__countvectorizer__min_df": [1, 2, 3],
    "header__columntransformer__pipeline-3__countvectorizer__ngram_range": [(1, 1), (1, 2)],
    
    "body__columntransformer__pipeline-4__countvectorizer__max_features": [5000, 10000, 15000, 20000, 25000, 30000],
    "body__columntransformer__pipeline-4__countvectorizer__max_df": [0.8, 0.9, 1.0],
    "body__columntransformer__pipeline-4__countvectorizer__min_df": [1, 2, 3],
    "body__columntransformer__pipeline-4__countvectorizer__ngram_range": [(1, 1), (1, 2)],
    
    "body__columntransformer__pipeline-5__countvectorizer__max_features": [5000, 10000, 15000, 20000, 25000, 30000],
    "body__columntransformer__pipeline-5__countvectorizer__max_df": [0.8, 0.9, 1.0],
    "body__columntransformer__pipeline-5__countvectorizer__min_df": [1, 2, 3],
    "body__columntransformer__pipeline-5__countvectorizer__ngram_range": [(1, 1), (1, 2)],

    "body__columntransformer__pipeline-6__countvectorizer__max_features": [5000, 10000, 15000, 20000, 25000, 30000],
    "body__columntransformer__pipeline-6__countvectorizer__max_df": [0.8, 0.9, 1.0],
    "body__columntransformer__pipeline-6__countvectorizer__min_df": [1, 2, 3],
    "body__columntransformer__pipeline-6__countvectorizer__ngram_range": [(1, 1), (1, 2)],

    "body__columntransformer__pipeline-7__countvectorizer__max_features": [5000, 10000, 15000, 20000, 25000, 30000],
    "body__columntransformer__pipeline-7__countvectorizer__max_df": [0.8, 0.9, 1.0],
    "body__columntransformer__pipeline-7__countvectorizer__min_df": [1, 2, 3],
    "body__columntransformer__pipeline-7__countvectorizer__ngram_range": [(1, 1), (1, 2)],

    "final_estimator__C": loguniform(1e-2, 1e2),
    "final_estimator__kernel": ['linear', 'rbf', 'poly'],
    "final_estimator__gamma": ['scale', 'auto']
}

random_search_stacking = RandomizedSearchCV(
    stacking,
    param_distributions=param_dist_stacking,
    n_iter=200,
    scoring="f1",
    cv=5,
    verbose=2,
    random_state=123,
    n_jobs=-1
)


print("Preprocessor and Stacking model created successfully.")
print("Randomized search started...")

random_search_stacking.fit(X_train, y_train_num)

print("Randomized search completed.")

stacking_best_param = random_search_stacking.best_params_
pickle.dump(stacking_best_param, open("ht_script_result.pkl", "wb"))

print("Parameters saved successfully.")

# Save the time taken as txt file
timetaken = datetime.now() - startTime
with open("ht_script_time.txt", "w") as f:
    f.write(str(timetaken))


