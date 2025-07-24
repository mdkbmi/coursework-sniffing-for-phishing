import os, sys
import click
import pandas as pd
import altair as alt

alt.data_transformers.enable('vegafusion')

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder
)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_validation import __validate_features_df
from src.model_evaluation import (
    mean_std_cross_val_scores, tidy_cv_results,
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
        raise ValueError(f"Validation for features_df failed: {e}")

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
    preprocessor_header = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', drop='if_binary'), header_bool_feats + header_cat_feats),
        (StandardScaler(), header_num_feats),
        remainder='drop',
    )

    preprocessor_subject = make_column_transformer(
        (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), header_text_feats),
        remainder='drop'    
    )

    preprocessor_header_combined = make_column_transformer(
        (
            OneHotEncoder(handle_unknown='ignore', drop='if_binary'), 
            header_bool_feats + header_cat_feats
        ),
        (
            StandardScaler(), 
            header_num_feats
        ),
        (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), header_text_feats),
        remainder='drop',
    )

    preprocessor_body = make_column_transformer(
        (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), body_text_feats),
        remainder='drop'
    )

    preprocessor_body_combined = make_column_transformer(
        (
            OneHotEncoder(handle_unknown='ignore', drop='if_binary'), 
            body_bool_feats + body_cat_feats
        ),
        (
            StandardScaler(), 
            body_num_feats
        ),
        (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), body_text_feats),
        remainder='drop',
    )

    preprocessor_body_nontext = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', drop='if_binary'), body_bool_feats + body_cat_feats),
        (StandardScaler(), body_num_feats),
        remainder='drop'
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
    pipe_header = make_pipeline(
        preprocessor_header,
        XGBClassifier(n_jobs=-1, objective="binary:logistic", enable_categorical=True)
    )

    pipe_subject = make_pipeline(
        preprocessor_subject,
        XGBClassifier(n_jobs=-1, objective="binary:logistic")
    )

    pipe_header_combined = make_pipeline(
        preprocessor_header_combined,
        XGBClassifier(objective="binary:logistic", enable_categorical=True)
    )

    pipe_body = make_pipeline(
        preprocessor_body,
        XGBClassifier(objective="binary:logistic")
    )

    pipe_body_nontext = make_pipeline(
        preprocessor_body_nontext,
        XGBClassifier(objective="binary:logistic", enable_categorical=True)
    )

    pipe_body_combined = make_pipeline(
        preprocessor_body_combined,
        XGBClassifier(objective="binary:logistic", enable_categorical=True)
    )

    pipe_all = make_pipeline(
        preprocessor_all,
        XGBClassifier(objective="binary:logistic", enable_categorical=True)
    )

    estimators = [
        ("header", pipe_header), 
        ("subject", pipe_subject), 
        ("body", pipe_body),
        ("body_nontext", pipe_body_nontext)
    ]

    estimators_combined = [
        ("header", pipe_header_combined), 
        ("body", pipe_body_combined),
    ]

    xgb = pipe_all

    sc_2 = StackingClassifier(
        estimators=estimators_combined,
        final_estimator=SVC(
            probability=True, 
            class_weight='balanced',
        ),
        n_jobs=-1,
    )

    sc_4 = StackingClassifier(
        estimators=estimators,
        final_estimator=SVC(
            probability=True, 
            class_weight='balanced',
        ),
        n_jobs=-1,
    )

    cv_results = {}
    models = {
        '1 XGBClassifier': xgb,
        '2 XGBClassifier + SVC': sc_2,
        '4 XGBClassifier + SVC': sc_4,
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
    os.makedirs(os.path.join(results_to, "model_architecture_selection"), exist_ok=True)
    cv_results_df.to_csv(os.path.join(results_to, "model_architecture_selection/cv_results.csv"), index=True)
    print('Saved results!')

    return

if __name__ == '__main__':
    main()
