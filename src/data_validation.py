# data_validation.py
# author: Danish Karlin Isa
# date: 2025-05-21

import pandas as pd
import pandera as pa
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Set, Literal
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

class EmailMetadata(BaseModel):
    From_email: Optional[str] = None
    From_email_domain: Optional[str] = None
    To_email: Optional[str] = None
    To_email_domain: Optional[str] = None
    Subject: str
    Received: Optional[List[str]] = None
    Authentication_Results: Optional[str] = Field(None, alias='Authentication-Results')
    Received_SPF: Optional[str] = Field(None, alias='Received-SPF')
    DKIM_Signature: Optional[str] = Field(None, alias='DKIM-Signature')
    Reply_To_domain: Optional[str] = Field(None, alias='Reply-To_domain')
    Content_Language: Optional[str] = Field(None, alias='Content-Language')
    Content_types: List[str]
    text_plain: str
    text_clean: str
    text_html: Optional[bytes] = None
    text_hyperlinks: Set[str]
    
    class Config:
        validate_by_name = True
        
    @field_validator('Received')
    def validate_received(cls, v):
        if v is None:
            return []
        return v
        
    @field_validator('text_hyperlinks')
    def validate_hyperlinks(cls, v):
        if v is None:
            return []
        return v

def __validate_metadata_df(metadata_df: pd.DataFrame) -> None:
    """
    Validate the structure and data types of metadata_df.
    To be used with `scripts/build_metadata_df.py`.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        The DataFrame (metadata_df) to validate.

    Returns
    -------
    None
        This function does not return anything. It will raise a SchemaError
        if the DataFrame does not conform to the expected schema.

    Raises
    ------
    pandera.errors.SchemaError
        If the DataFrame does not match the expected schema definition.
    """
    schema_metadata_df = pa.DataFrameSchema(
        {
            "From_email": pa.Column(str, nullable=True,),
            "From_email_domain": pa.Column(str, nullable=True,),
            "To_email": pa.Column(str, nullable=True,),
            "To_email_domain": pa.Column(str, nullable=True,),
            "Subject": pa.Column(str, nullable=True),
            "Received": pa.Column(
                object, nullable=True
            ),
            "Authentication-Results": pa.Column(str, nullable=True,),
            "Received-SPF": pa.Column(str, nullable=True,),
            "DKIM-Signature": pa.Column(str, nullable=True,),
            "Content-Language": pa.Column(str, nullable=True,),
            "Reply-To_domain": pa.Column(str, nullable=True,),
            "Content_types": pa.Column(object, nullable=True),
            "text_html": pa.Column(bytes, nullable=True),
            "text_plain": pa.Column(str),
            "text_clean": pa.Column(str),
            "text_hyperlinks": pa.Column(object, nullable=True),

            "target_1": pa.Column(
                pd.CategoricalDtype(['benign', 'malicious'], True),
                nullable=False
            ),
            "target_2": pa.Column(
                pd.CategoricalDtype(['ceo_fraud', 'phishing', 'reply-chain-attack', 'legitimate', 'spam'], False),
                nullable=False
            ),
            "target_3": pa.Column(
                pd.CategoricalDtype([
                    'gift_cards', 'payroll_update', 'wire_transfers', 'third_party', 'outbound',
                    'ubc', 'ubc_outbound', 'spearphishing', 'reply-chain-attack',
                    'legitimate_email_confirmed', 'spam_false_positive', 'inbound'
                    ], False),
                nullable=False
            ),
        }
    )
    try:
        schema_metadata_df.validate(metadata_df)
    except Exception as e:
        print(f'Errors occurred during validation of metadata_df: {e}')
        raise e

class EmailFeatures(BaseModel):
    subject: str
    url_present_in_subject: bool
    routing_length_before_ubc: int = Field(ge=0)
    dmarc_authentication_present: bool
    dkim_sender_domains_match: bool
    to_from_addresses_match: bool
    sender_email_spf_match: bool
    different_reply_domains: bool
    internal_server_transfer_count: int = Field(ge=0)
    name_server_match: bool
    dkim_result: str
    spf_result: str
    dmarc_result: str
    
    text_clean: str
    word_count: int = Field(ge=0)
    readable_proportion: float = Field(ge=0, le=1)
    whitespace_ratio: float = Field(ge=0, le=1)
    alphabet_proportion: float = Field(ge=0, le=1)
    grammar_error_rate: float = Field(ge=0, le=1)
    english_french_proportion: float = Field(ge=0, le=1)
    text_content_count: int = Field(ge=0)
    multimedia_content_count: int = Field(ge=0)
    others_content_count: int = Field(ge=0)
    hyperlink_proportion: float = Field(ge=0, le=1)
    
    non_ascii_present: bool
    hidden_text_present: bool
    empty_body: bool
    html_parsing_error: Literal[-1, 0, 1]


def __validate_features_df(features_df):
    """
    Validates the DataFrame structure and types for the email features data.
    To be used with `scripts/build_features_df.py`.

    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame containing extracted email features to validate

    Returns
    -------
    features_df : pandas.DataFrame
        The validated DataFrame if validation passes

    Raises
    ------
    pandera.errors.SchemaError
        If the DataFrame does not match the expected schema
    """

    schema_features_df = pa.DataFrameSchema({
        "subject": pa.Column(str, nullable=True),
        "url_present_in_subject": pa.Column(bool),
        "routing_length_before_ubc": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
        "dmarc_authentication_present": pa.Column(bool),
        "dkim_sender_domains_match": pa.Column(bool),
        "to_from_addresses_match": pa.Column(bool),
        "sender_email_spf_match": pa.Column(bool),
        "different_reply_domains": pa.Column(bool),
        "internal_server_transfer_count": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
        "name_server_match": pa.Column(bool),
        "dkim_result": pa.Column(str),
        "spf_result": pa.Column(str),
        "dmarc_result": pa.Column(str),

        "text_clean": pa.Column(str, nullable=True),
        "word_count": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
        "readable_proportion": pa.Column(float, pa.Check.in_range(0, 1)),
        "whitespace_ratio": pa.Column(float, pa.Check.in_range(0, 1)),
        "alphabet_proportion": pa.Column(float, pa.Check.in_range(0, 1)),
        "grammar_error_rate": pa.Column(float, pa.Check.in_range(0, 1)),
        "english_french_proportion": pa.Column(float, pa.Check.in_range(0, 1)),
        "text_content_count": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
        "multimedia_content_count": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
        "others_content_count": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
        "hyperlink_proportion": pa.Column(float, pa.Check.in_range(0, 1)),

        "non_ascii_present": pa.Column(bool),
        "hidden_text_present": pa.Column(bool),
        "empty_body": pa.Column(bool),
        "html_parsing_error": pa.Column(int, pa.Check.isin([-1, 0, 1])),

        "target_1": pa.Column(
            pd.CategoricalDtype(['benign', 'malicious'], True),
            nullable=False
        ),
        "target_2": pa.Column(
            pd.CategoricalDtype(['ceo_fraud', 'phishing', 'reply-chain-attack', 'legitimate', 'spam'], False),
            nullable=False
        ),
        "target_3": pa.Column(
            pd.CategoricalDtype([
                'gift_cards', 'payroll_update', 'wire_transfers', 'third_party', 'outbound',
                'ubc', 'ubc_outbound', 'spearphishing', 'reply-chain-attack',
                'legitimate_email_confirmed', 'spam_false_positive', 'inbound'
                ], False),
            nullable=False
        ),
    })

    schema_features_df.validate(features_df)

class ProbabilityScores(BaseModel):
    benign: float = Field(ge=0.0, le=1.0, description="Probability score for benign class")
    malicious: float = Field(ge=0.0, le=1.0, description="Probability score for malicious class")
    
    @field_validator('benign', 'malicious')
    def validate_probability(cls, v):
        if not (0 <= v <= 1):
            raise ValueError(f"Probability must be between 0 and 1, got {v}")
        return float(v)


class PredictionResult(BaseModel):
    probability: ProbabilityScores = Field(
        description="Probability scores for each class"
    )
    
    @field_validator('probability')
    def validate_probabilities_sum(cls, v):
        probability_sum = v.benign + v.malicious
        if not (0.99 <= probability_sum <= 1.01):
            raise ValueError(f"Probability scores should sum to approximately 1.0, got {probability_sum}")
        return v
    
def __validate_model(model: StackingClassifier):
    if not isinstance(model, StackingClassifier):
        raise ValueError(f"Expected model to be of type sklearn.ensemble._stacking.StackingClassifier, not {type(model)}.")
    
    expected_estimators = ['header', 'subject', 'body', 'body_nontext']
    if list(model.named_estimators.keys()) != expected_estimators:
        raise ValueError(f"Model does not contain the right estimators")
    
    if not isinstance(model.final_estimator_, SVC):
        raise ValueError(f"Expected final estimator to be of type sklearn.svm._classes.SVC, not {type(model.final_estimator_)}")
    
    try:
        check_is_fitted(model)
    except Exception as e:
        raise ValueError("Model is not fitted yet.")
    
    return

class ProbabilityScores(BaseModel):
    benign: float = Field(ge=0.0, le=1.0, description="Probability score for benign class")
    malicious: float = Field(ge=0.0, le=1.0, description="Probability score for malicious class")
    
    @field_validator('benign', 'malicious')
    def validate_probability(cls, v):
        if not (0 <= v <= 1):
            raise ValueError(f"Probability must be between 0 and 1, got {v}")
        return float(v)


class PredictionResult(BaseModel):
    probability: ProbabilityScores = Field(
        description="Probability scores for each class"
    )
    
    @field_validator('probability')
    def validate_probabilities_sum(cls, v):
        probability_sum = v.benign + v.malicious
        if not (0.99 <= probability_sum <= 1.01):
            raise ValueError(f"Probability scores should sum to approximately 1.0, got {probability_sum}")
        return v
    
def __validate_model(model: StackingClassifier):
    if not isinstance(model, StackingClassifier):
        raise ValueError(f"Expected model to be of type sklearn.ensemble._stacking.StackingClassifier, not {type(model)}.")
    
    expected_estimators = ['header', 'subject', 'body', 'body_nontext']
    if list(model.named_estimators.keys()) != expected_estimators:
        raise ValueError(f"Model does not contain the right estimators")
    
    if not isinstance(model.final_estimator_, SVC):
        raise ValueError(f"Expected final estimator to be of type sklearn.svm._classes.SVC, not {type(model.final_estimator_)}")
    
    try:
        check_is_fitted(model)
    except Exception as e:
        raise ValueError("Model is not fitted yet.")
    
    return

class ProbabilityScores(BaseModel):
    benign: float = Field(ge=0.0, le=1.0, description="Probability score for benign class")
    malicious: float = Field(ge=0.0, le=1.0, description="Probability score for malicious class")
    
    @field_validator('benign', 'malicious')
    def validate_probability(cls, v):
        if not (0 <= v <= 1):
            raise ValueError(f"Probability must be between 0 and 1, got {v}")
        return float(v)


class PredictionResult(BaseModel):
    probability: ProbabilityScores = Field(
        description="Probability scores for each class"
    )
    
    @field_validator('probability')
    def validate_probabilities_sum(cls, v):
        probability_sum = v.benign + v.malicious
        if not (0.99 <= probability_sum <= 1.01):
            raise ValueError(f"Probability scores should sum to approximately 1.0, got {probability_sum}")
        return v