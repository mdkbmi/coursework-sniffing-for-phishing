"""
PhishSense v2 Model Training Script

This script trains a stacked ensemble model to classify phishing emails using header, subject, 
and body features. It supports optional hyperparameter tuning for text vectorization components.
The model consists of four component classifiers (header metadata, subject text, body text, 
and body metadata) combined using a stacking approach with SVC as the meta-classifier.
Features are processed using appropriate transformers (OneHotEncoder for categorical features,
StandardScaler for numerical features, and CountVectorizer for text features).

Usage:
    python train_model.py --train_data PATH_TO_FEATURES --model_to SAVE_DIRECTORY --tuning [y/n]

Input:
    - Parquet file containing email features and target labels
    
Output:
    - Trained model saved as a pickle file (phishsense-v2.pkl)
"""

import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

import pickle
import os, sys
import click

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_validation import __validate_features_df

@click.command()
@click.option('--train_data', type=str, required=True, help="Path to training data (features_df)")
@click.option('--model_to', type=str, required=True, help="Directory to save the trained model in")
@click.option('--tuning', type=str, required=True, help="Include hyperparameter tuning for CountVectorizer? [y/n]")
def main(train_data: str, model_to: str, tuning: str) -> None:

    if not os.path.exists(train_data):
        raise FileNotFoundError(f"{train_data} is not valid.")
    
    if not train_data.endswith('.parquet'):
        raise ValueError(f"{train_data} is not the expected file.")
    
    if not os.path.exists(model_to):
        raise FileNotFoundError(f"Directory for model_path '{model_to}' does not exist.")
    
    try:
        test_file_path = os.path.join(model_to, "write_permission_test")
        with open(test_file_path, 'w') as f:
            f.write("test")
        os.remove(test_file_path)
    except Exception as e:
        raise PermissionError(f"No write permission for the model directory: {e}")
    model_path = os.path.join(model_to, "phishsense-v2.pkl")
    
    if tuning not in ['y', 'n']:
        raise ValueError(f"tuning accepts either `y`, `n` or blank, not {tuning}.")
    
    train_df = pd.read_parquet(train_data)
    print('Successfully loaded training data!')

    try:
        __validate_features_df(train_df)
    except Exception as e:
        raise ValueError(f"Validation for train_data failed: {e}")

    train_df = train_df.copy()
    X_train = train_df.drop(columns=['target_1', 'target_2', 'target_3'])
    y_train = train_df['target_1']

    header_text_feats = 'subject'

    header_bool_feats = [
        'url_present_in_subject', 
        'dmarc_authentication_present',
        'dkim_sender_domains_match',
        'to_from_addresses_match', 
        'sender_email_spf_match',
        'different_reply_domains',
        'name_server_match', 
    ]

    header_cat_feats = [
        'dkim_result',
        'spf_result',
        'dmarc_result',
    ]

    header_num_feats = [
        'routing_length_before_ubc',
        'internal_server_transfer_count',
    ]

    body_text_feats = 'text_clean'

    body_bool_feats = [
        'non_ascii_present',
        'hidden_text_present',
        'empty_body',
    ]

    body_cat_feats = [
        'html_parsing_error',
    ]

    body_num_feats = [
        'word_count',
        'readable_proportion',
        'whitespace_ratio',
        'alphabet_proportion',
        'grammar_error_rate',
        'english_french_proportion',
        'text_content_count',
        'multimedia_content_count',
        'others_content_count',
        'hyperlink_proportion',
    ]

    scoring = make_scorer(f1_score, pos_label='benign')

    param_grid = {
        'subject__columntransformer__countvectorizer__min_df': [0.001, 0.01, 0.05],
        'subject__columntransformer__countvectorizer__max_df': [0.95, 0.99, 0.999],
        'subject__columntransformer__countvectorizer__max_features': [500, 1000, 10000],
        'body__columntransformer__countvectorizer__min_df': [0.001, 0.01, 0.05],
        'body__columntransformer__countvectorizer__max_df': [0.95, 0.99, 0.999],
        'body__columntransformer__countvectorizer__max_features': [1000, 10000, 100000],
    }

    subject_countvec_min_df = 0.001
    subject_countvec_max_df = 0.99
    subject_countvec_max_features = 500

    body_countvec_min_df = 0.001
    body_countvec_max_df = 0.99
    body_countvec_max_features = 10000

    preprocessor_header = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', drop='if_binary'), header_bool_feats + header_cat_feats),
        (StandardScaler(), header_num_feats),
        remainder='drop',
    )

    preprocessor_subject = make_column_transformer(
        (
            CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french'],
                            min_df=subject_countvec_min_df, max_df=subject_countvec_max_df,
                            max_features=subject_countvec_max_features), 
            header_text_feats
        ),
        remainder='drop'    
    )

    preprocessor_body = make_column_transformer(
        (
            CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french'],
                            min_df=body_countvec_min_df, max_df=body_countvec_max_df,
                            max_features=body_countvec_max_features), 
            body_text_feats
        ),
        remainder='drop'    
    )

    preprocessor_body_nontext = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', drop='if_binary'), body_bool_feats + body_cat_feats),
        (StandardScaler(), body_num_feats),
        remainder='drop'
    )

    pipe_header = make_pipeline(
        preprocessor_header,
        XGBClassifier(objective="binary:logistic", enable_categorical=True)
    )

    pipe_subject = make_pipeline(
        preprocessor_subject,
        XGBClassifier(objective="binary:logistic")
    )

    pipe_body = make_pipeline(
        preprocessor_body,
        XGBClassifier(objective="binary:logistic")
    )

    pipe_body_nontext = make_pipeline(
        preprocessor_body_nontext,
        XGBClassifier(objective="binary:logistic", enable_categorical=True)
    )

    estimators = [
        ("header", pipe_header), 
        ("subject", pipe_subject), 
        ("body", pipe_body),
        ("body_nontext", pipe_body_nontext)
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=SVC(
            probability=True, 
            class_weight='balanced',
        ),
        n_jobs=-1,
    )

    rs = RandomizedSearchCV(
        model, param_grid, n_jobs=-1, cv=5, return_train_score=True, 
        scoring=scoring, n_iter=200, verbose=3
    )

    print('Successfully initialised model object!')

    if tuning == 'y':
        print('Tuning selected!')
        print('WARNING: Tuning is a time-consuming process and may take hours.')
        print('If you mistakenly opted for tuning, please abort script.')
        rs.fit(X_train, y_train)

        with open(model_path, 'wb') as f:
            pickle.dump(rs.best_estimator_, f)

    else:
        model.fit(X_train, y_train)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print("Model successfully trained!")

    return

if __name__ == "__main__":
    main()