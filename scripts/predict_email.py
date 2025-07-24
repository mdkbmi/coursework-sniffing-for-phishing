"""
Email Prediction Script

This script predicts email characteristics based on machine learning models.
It processes an email file through the following pipeline:
1. Extracts metadata from the email
2. Generates features from the metadata
3. Makes predictions using a specified model
4. Validates data at each step using Pydantic models

Usage:
    python predict_email.py --email_path PATH_TO_EMAIL --model_path PATH_TO_MODEL --show [y/n]

Arguments:
    --email_path: Path to the email file to analyze
    --model_path: Path to the trained model file (pickle format)
    --show: Whether to print the prediction results (y/n)
    
Returns:
    A dictionary containing prediction results
"""

import os, sys
import click
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.metadata_extraction import extract_email_metadata
from src.feature_generation import generate_features_from_metadata
from src.prediction import generate_prediction
from src.data_validation import EmailMetadata, EmailFeatures, PredictionResult
    
@click.command()
@click.option('--email_path', type=str, required=True, help="Path to email file")
@click.option('--model_path', type=str, required=True, help="Path to model pickle file")
@click.option('--show', type=str, required=True, help="Print predictions? [y/n]")
def main(email_path: str, model_path: str, show: str) -> dict:
    paths = [email_path, model_path]
    
    for path in paths:
        if not isinstance(path, str):
            raise TypeError(f"{path} must be a string.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist.")
    
    if show is not None and show.lower() not in ['y', 'n']:
        raise ValueError("--print only accepts y, n")
    
    metadata = extract_email_metadata(email_path)
    if isinstance(metadata, dict):
        try:
            EmailMetadata.model_validate(metadata)
        except Exception as e:
            raise ValueError("Errors occurred during validation of `metadata`: {e}")
    else:
        raise TypeError("`metadata` should be a dict.")
    
    features = generate_features_from_metadata(metadata)
    if isinstance(features, dict):
        try:
            EmailFeatures.model_validate(features)
        except Exception as e:
            raise ValueError("Errors occurred during validation of `features`: {e}")
    else:
        raise TypeError("`features` should be a dict.")

    predictions = generate_prediction(features, model_path)
    try:
        PredictionResult.model_validate(predictions)
    except Exception as e:
        raise ValueError(f"Prediction validation failed: {e}")

    if show is not None and show.lower() == 'y':
        print(json.dumps(predictions))

    return predictions

if __name__ == '__main__':
    main()
