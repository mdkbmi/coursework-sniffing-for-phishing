import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import pickle

original_df = pd.read_parquet(
    '/data/workspace/danishki/git_repo/data/full-dataset/raw/train.parquet'
).query(
    '`target_3` != "self_phishing"'
)

features_df = pd.read_parquet(
    '/data/workspace/danishki/git_repo/data/full-dataset/processed/train.parquet'
)
features_df = features_df.copy()

features_df.loc[features_df['empty_body'] == True, 'target_1'] = 'malicious'

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

preprocessor_header = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', drop='if_binary'), header_bool_feats + header_cat_feats),
    (StandardScaler(), header_num_feats),
    remainder='drop',
)

preprocessor_subject = make_column_transformer(
    (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), header_text_feats),
    remainder='drop'    
)

preprocessor_body = make_column_transformer(
    (CountVectorizer(strip_accents='unicode', lowercase=True, stop_words=['english', 'french']), body_text_feats),
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

svc = StackingClassifier(
    estimators=estimators,
    final_estimator=SVC(
        probability=True, 
        class_weight='balanced',
    ),
    verbose=3
)

scoring = make_scorer(f1_score, pos_label='benign')

# param_grid = {
#     'subject__columntransformer__countvectorizer__min_df': [0.001, 0.01, 0.05],
#     'subject__columntransformer__countvectorizer__max_df': [0.95, 0.99, 0.999],
#     'subject__columntransformer__countvectorizer__max_features': [500, 1000, 10000],
#     'body__columntransformer__countvectorizer__min_df': [0.001, 0.01, 0.05],
#     'body__columntransformer__countvectorizer__max_df': [0.95, 0.99, 0.999],
#     'body__columntransformer__countvectorizer__max_features': [1000, 10000, 100000],
# }

param_grid = {
    'header__xgbclassifier__reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
    'header__xgbclassifier__reg_lambda': [0.01, 0.1, 1.0, 10.0],
    'subject__xgbclassifier__reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
    'subject__xgbclassifier__reg_lambda': [0.01, 0.1, 1.0, 10.0],
    'body__xgbclassifier__reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
    'body__xgbclassifier__reg_lambda': [0.01, 0.1, 1.0, 10.0],
    'body_nontext__xgbclassifier__reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
    'body_nontext__xgbclassifier__reg_lambda': [0.01, 0.1, 1.0, 10.0],
}

rs = RandomizedSearchCV(
    svc, param_grid, n_jobs=-1, cv=5, return_train_score=True, 
    scoring=scoring, n_iter=200, verbose=3
)

rs.fit(X_train, y_train)

with open('/data/workspace/danishki/git_repo/notebooks/milestone5/rscv-xgb-reg.pkl', 'wb') as file:
    pickle.dump(rs, file)
