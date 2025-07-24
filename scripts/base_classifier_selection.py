import os, sys
import click
import pandas as pd
import numpy as np
import altair as alt

alt.data_transformers.enable('vegafusion')

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, FunctionTransformer,
    LabelEncoder
)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_validation import __validate_features_df
from src.model_evaluation import (
    mean_std_cross_val_scores, scorer, tidy_cv_results,
    get_xgb_feature_importances, generate_feature_importances_chart
)

@click.command()
@click.option('--train_data', type=str, required=True, help="Path to features_df (parquet)")
@click.option('--results_to', type=str, required=True, help="Directory to save results in")
def main(train_data: str, results_to: str):
    if not os.path.exists(train_data):
        raise FileNotFoundError(f"features_df_path does not exist: {train_data}")
    
    if not train_data.endswith(".parquet"):
        raise TypeError(f"Expected parquet file")
    
    try:
        test_file_path = os.path.join(results_to, "write_permission_test")
        with open(test_file_path, 'w') as f:
            f.write("test")
        os.remove(test_file_path)
    except Exception as e:
        raise PermissionError(f"No write permission for the results directory: {e}")

    print('Reading in features_df...')
    features_df = pd.read_parquet(train_data)

    try:
        __validate_features_df(features_df)
    except Exception as e:
        print(f"Validation for features_df failed: {e}")

    print('Performing train-test split...')
    X_train, X_test, y_train, y_test = train_test_split(
        features_df.drop(columns=['target_1', 'target_2', 'target_3']), features_df['target_1'],
        train_size=0.7, random_state=42
    )
    
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

    for feat in header_cat_feats + body_cat_feats:
        features_df[feat] = pd.Categorical(features_df[feat])

    print('Setting up preprocessors...')
    preprocessor_text = make_column_transformer(
        (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), header_text_feats),
        (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), body_text_feats),
        remainder='drop',
    )

    preprocessor_all = make_column_transformer(
        (
            OneHotEncoder(handle_unknown='ignore', drop='if_binary'), 
            header_bool_feats + header_cat_feats + body_bool_feats + body_cat_feats
        ),
        (
            StandardScaler(), 
            header_num_feats + body_num_feats
        ),
        (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), header_text_feats),
        (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), body_text_feats),
        remainder='drop',
    )

    print('Initialising models...')
    dummy = make_pipeline(
        preprocessor_all,
        DummyClassifier(random_state=42)
    )

    lr = make_pipeline(
        preprocessor_all,
        LogisticRegression()
    )

    nb = make_pipeline(
        preprocessor_text,
        FunctionTransformer(lambda x: x.toarray(), accept_sparse=True),
        GaussianNB()
    )

    svc = make_pipeline(
        preprocessor_all,
        SVC(random_state=42)
    )

    rf = make_pipeline(
        preprocessor_all,
        RandomForestClassifier(random_state=42)
    )

    xgb = make_pipeline(
        preprocessor_all,
        XGBClassifier(n_jobs=-1)
    )

    cv_results = {}
    models = {
        'DummyClassifier': dummy,
        'LogisticRegression': lr,
        'GaussianNB': nb,
        'SVC': svc,
        'RandomForestClassifier': rf,
        'XGBClassifier': xgb,
    }

    le = LabelEncoder()
    le.fit(y_train)

    for name, model in models.items():
        print(f'Performing cross-validation for {name}...')
        cv_results[name] = mean_std_cross_val_scores(
            model, X_train, y_train
        )

    print('Cross-validation complete!')
    print('Formatting results...')
    cv_results_df = tidy_cv_results(cv_results)
    os.makedirs(os.path.join(results_to, "base_classifier_selection"), exist_ok=True)
    cv_results_df.to_csv(os.path.join(results_to, "base_classifier_selection/cv_results.csv"), index=True)
    print('Saved results!')

    print('Generating feature importances for XGBClassifier...')
    model = models['XGBClassifier']
    model.fit(X_train, le.transform(y_train))

    feature_importances = get_xgb_feature_importances(
        model.named_steps['xgbclassifier'],
        model.named_steps['columntransformer']
    )
    
    feature_importances.to_csv(os.path.join(results_to, "base_classifier_selection/xgb_feature_importances.csv"), index=True)
    print('Saved results!')

    print('Rendering feature importances chart...')
    feature_importances_chart = generate_feature_importances_chart(
        feature_importances, 10, "Top 10 feature importances in base XGBClassifier"
    )

    feature_importances_chart.save(
        os.path.join(results_to, "base_classifier_selection/xgb_feature_importances.png"),
        ppi=300
    )

    print('Saved chart!')

    return

if __name__ == '__main__':
    main()
