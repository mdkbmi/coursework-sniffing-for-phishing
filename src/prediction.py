import os, sys
import pickle
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_validation import EmailFeatures, PredictionResult, __validate_model

def generate_prediction(features: dict, model_path: str) -> dict:
    """
    Generate a prediction for email classification using a pre-trained model.

    Parameters
    ----------
    features : dict
        A dictionary containing the email features to be classified.
        Must be validatable by EmailFeatures.

    Returns
    -------
    dict
        A dictionary containing:
        - 'probability' : dict
            A dictionary with probabilities for each class:
            - 'benign' : float
            - 'malicious' : float

    Raises
    ------
    ValueError
        If the features validation fails or if the model returns an unexpected label.

    Notes
    -----
    The function loads a pre-trained model from a specific file path and uses it
    to classify the given email features.
    """

    try:
        EmailFeatures.model_validate(features)
    except Exception as e:
        raise ValueError(f"Validation for features failed: {e}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    try:
        __validate_model(model)
    except Exception as e:
        raise ValueError(f"Model validation failed: {e}")
    
    input_df = pd.DataFrame(features, index=[0])
    proba = model.predict_proba(input_df)[0]

    results = {
        'probability': {
            'benign': float(proba[0]),
            'malicious': float(proba[1])
        }
    }
    
    try:
        PredictionResult.model_validate(results)
        return results
    except Exception as e:
        raise ValueError(f"Prediction validation failed: {e}")
