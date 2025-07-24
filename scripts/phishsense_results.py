# generate_phishsense_metrics.py
# author: Joseph Lim
# date: 2025-06-18

import os
import warnings
import pandas as pd
import json
import click

@click.command()
@click.command('--predictions' , type=str, required=True, help="PhishSense predictions in JSON format")
@click.option('--results_to', type=str, required=True, help="Directory to save the results in")
def main(
    results_to: str,
    input_json="/data/workspace/dataset/phishsense/phishsense_output.json"
):
    # Read in the phishsense output JSON file
    with open(input_json, "r") as f:
        phishsense_prediction = json.load(f)

    rows = []
    for label, emails in phishsense_prediction.items():
        for path, data in emails.items():
            row = {
                "path": path,
                "body_ceo_fraud": data["body"].get("CEO Fraud", 0),
                "body_legitimate": data["body"].get("Legitimate", 0),
                "body_phishing": data["body"].get("Phishing", 0),
                "body_spam": data["body"].get("Spam", 0),
                "subject_ceo_fraud": data["subject"].get("CEO Fraud", 0),
                "subject_legitimate": data["subject"].get("Legitimate", 0),
                "subject_phishing": data["subject"].get("Phishing", 0),
                "subject_spam": data["subject"].get("Spam", 0),
                "label": label
            }
            rows.append(row)

    phishsense_pred_df = pd.DataFrame(rows)

    body_map = {
        'body_legitimate': 'benign',
        'body_spam': 'benign',
        'body_ceo_fraud': 'malicious',
        'body_phishing': 'malicious'
    }
    subject_map = {
        'subject_legitimate': 'benign',
        'subject_spam': 'benign',
        'subject_ceo_fraud': 'malicious',
        'subject_phishing': 'malicious'
    }

    phishsense_pred_df['body_prediction'] = (
        phishsense_pred_df[['body_ceo_fraud', 'body_legitimate', 'body_phishing', 'body_spam']]
        .idxmax(axis=1)
        .replace(body_map)
    )
    phishsense_pred_df['subject_prediction'] = (
        phishsense_pred_df[['subject_ceo_fraud', 'subject_legitimate', 'subject_phishing', 'subject_spam']]
        .idxmax(axis=1)
        .replace(subject_map)
    )

    body_pred_df = phishsense_pred_df[['path', 'label', 'body_prediction']].copy()

    y_true_body = body_pred_df['label']
    y_pred_body = body_pred_df['body_prediction']

    body_results_df = pd.DataFrame({
        'path': body_pred_df['path'],
        'true': y_true_body,
        'pred': y_pred_body
    })

    results_output_path = os.path.join(os.path.dirname(results_to), 'phishsense_predictions.csv')
    body_results_df.to_csv(results_output_path, index=False)

if __name__ == "__main__":
    main()
