import os, sys
import click
import pickle
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_validation import __validate_features_df, __validate_model
from src.model_evaluation import get_model_metrics

@click.command()
@click.option('--train_data', type=str, required=True, help="Path to train data")
@click.option('--test_data', type=str, required=True, help="Path to train data")
@click.option('--phishsense_results', type=str, required=True, help="Path to PhishSense results")
@click.option('--model_path', type=str, required=True, help="Path to model pickle")
@click.option('--results_to', type=str, required=True, help="Directory to save results in")
def main(train_data: str, test_data: str, phishsense_results: str, model_path: str, results_to: str):
    for path in [train_data, test_data, phishsense_results, model_path, results_to]:
        if not isinstance(path, str):
            raise TypeError(f"All arguments must be a string: {path}")
    
        if not os.path.exists(path):
            raise FileNotFoundError(f"File/directory not found: {path}")
        
    train_df = pd.read_parquet(train_data)
    try:
        __validate_features_df(train_df)
    except Exception as e:
        raise ValueError(f"Validation failed for {train_data}: {e}")

    test_df = pd.read_parquet(test_data)
    try:
        __validate_features_df(test_df)
    except Exception as e:
        raise ValueError(f"Validation failed for {test_data}: {e}")
    
    phishsense_df = pd.read_csv(phishsense_results)

    expected_columns = ['path', 'true', 'pred']
    missing_columns = [col for col in expected_columns if col not in phishsense_df.columns]
    if missing_columns:
        raise ValueError(f"PhishSense results missing required columns. Expected: {expected_columns}, Got: {list(phishsense_df.columns)}. Missing: {missing_columns}")
    
    if len(phishsense_df) != len(test_df):
        raise ValueError(f"PhishSense results length ({len(phishsense_df)}) doesn't match test data length ({len(test_df)})")

    print("Successfully loaded train and test data!")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    try:
        __validate_model(model)
    except Exception as e:
        raise ValueError(f"Model validation failed: {e}")
    
    print("Successfully loaded trained model!")
    
    X_train, y_train = train_df.drop(columns=['target_1', 'target_2', 'target_3']), train_df['target_1']
    X_test, y_test = test_df.drop(columns=['target_1', 'target_2', 'target_3']), test_df['target_1']
    
    y_train_proba = model.predict_proba(X_train)[:, 0]
    y_test_proba = model.predict_proba(X_test)[:, 0]

    y_train_pred = ['benign' if prob > 0.7 else 'malicious' for prob in y_train_proba]
    y_test_pred = ['benign' if prob > 0.7 else 'malicious' for prob in y_test_proba]

    print("Successfully generated predictions!")
    print("Generating classification reports...")

    train_results = get_model_metrics(y_train, y_train_pred)
    train_results['model'] = 'PhishSense-v2'
    train_results['dataset'] = 'train'

    test_results = get_model_metrics(y_test, y_test_pred)
    test_results['model'] = 'PhishSense-v2'
    test_results['dataset'] = 'test'

    phishsense = get_model_metrics(phishsense_df['true'], phishsense_df['pred'])
    phishsense['model'] = 'PhishSense'
    phishsense['dataset'] = 'test'


    results = {0: train_results, 
               1: test_results,
               2: phishsense}

    results_df = pd.DataFrame(results).T

    tidy_df = pd.DataFrame({
        'Model': results_df['model'],
        'Dataset': results_df['dataset'],
        'Precision': results_df['precision'],
        'Recall': results_df['recall'],
        'F1-score': results_df['f1-score'],
        'False Benign Rate / FNR': results_df['false_benign_rate'],
        'False Malicious Rate / FPR': results_df['false_malicious_rate']
    })

    os.makedirs(os.path.join(results_to, "final_results"), exist_ok=True)
    tidy_df.to_csv(os.path.join(results_to, "final_results/classification_report.csv"), index=True)

    print(f"Successfully saved classification reports to {results_to}")

    cm_train = ConfusionMatrixDisplay.from_predictions(
        y_train, 
        y_train_pred,
    )
    cm_train.ax_.set_title('PhishSense-v2 (Train)')
    
    cm_test = ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_test_pred,
    )
    cm_test.ax_.set_title('PhishSense-v2 (Test)')

    cm_phishsense = ConfusionMatrixDisplay.from_predictions(
        phishsense_df['true'], 
        phishsense_df['pred']
    )
    cm_phishsense.ax_.set_title('PhishSense (Test)')

    cm_train.figure_.savefig(os.path.join(results_to, "final_results/cm_phishsense-v2_train.png"), dpi=300)
    cm_test.figure_.savefig(os.path.join(results_to, "final_results/cm_phishsense-v2_test.png"), dpi=300)
    cm_phishsense.figure_.savefig(os.path.join(results_to, "final_results/cm_phishsense_test.png"), dpi=300)

if __name__ == "__main__":
    main()
