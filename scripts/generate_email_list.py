"""
Generate Email Lists for Classification Dataset

This script processes a collection of email (.eml) files organized in folders by category and prepares 
datasets for email classification tasks. It:

1. Maps folder names to hierarchical categories (binary_label/category/subcategory)
2. Collects paths to all .eml files with corresponding labels
3. Splits data into train (70%) and test (30%) sets with stratification
4. Creates two sampled versions of the training set:
    - Small (200 emails)
    - Large (3000 emails)

Output:
- Full dataset: ./data/full-dataset/train.csv, ./data/full-dataset/test.csv
- Sampled datasets: ./data/sampled-dataset/sample-small.csv, ./data/sampled-dataset/sample-large.csv

Each CSV contains columns:
- path: File path to the .eml file
- target_1: Binary label (malicious/benign)
- target_2: Category (ceo_fraud, phishing, legitimate, spam, etc.)
- target_3: Subcategory (more specific classification)
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main() -> None:
    # Mapping of subcategory folder to category and binary label
    label_map = {
        # Malicious
        "CEO_Fraud_-_Gift_Cards": ("gift_cards", "ceo_fraud", "malicious"),
        "CEO_Fraud_-_Payroll_Update": ("payroll_update", "ceo_fraud", "malicious"),
        "CEO_Fraud_-_Wire_Transfers": ("wire_transfers", "ceo_fraud", "malicious"),
        "Phishing_-_3rd_Party": ("third_party", "phishing", "malicious"),
        "Phishing_-_Outbound": ("outbound", "phishing", "malicious"),
        "Phishing_–_UBC": ("ubc", "phishing", "malicious"),
        "Phishing_UBC_-_Outbound": ("ubc_outbound", "phishing", "malicious"),
        "Self-Phishing": ("self_phishing", "phishing", "malicious"),
        "Spearphishing": ("spearphishing", "phishing", "malicious"),
        "Reply_Chain_Attack": ("reply-chain-attack", "reply-chain-attack", "malicious"),

        # Benign
        "Legitimate_Email_Confirmed": ("legitimate_email_confirmed", "legitimate", "benign"),
        "Spam_-_False_Positives": ("spam_false_positive", "legitimate", "benign"),
        "Spam_–_Inbound": ("inbound", "spam", "benign"),
        "Spam_–_Outbound": ("outbound", "spam", "benign"),
    }

    dataset_root = Path("/data/dataset")

    # Collect all .eml file entries
    rows = []
    for subfolder, (subcategory, category, binary_label) in label_map.items():
        eml_files = (dataset_root / subfolder).rglob("*.eml")
        for eml in eml_files:
            rel_path = eml.relative_to("/") 
            rows.append({
                "path": f"/{rel_path.as_posix()}",
                "target_1": binary_label,
                "target_2": category,
                "target_3": subcategory
            })

    # Build full DataFrame
    df = pd.DataFrame(rows)

    # Split into train and test sets
    # Stratified split on target_3 to preserve subcategory distribution
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["target_3"],
        random_state=42,
    )

    # Create output folder
    output_dir = Path("./data/full-dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Downsample train.csv
    # Sample small (200 rows)
    sample_small, _ = train_test_split(
        train_df,
        train_size=200,
        stratify=train_df["target_3"],
        random_state=42
    )

    # Sample large (3000 rows)
    sample_large, _ = train_test_split(
        train_df,
        train_size=3000,
        stratify=train_df["target_3"],
        random_state=42
    )

    # Output directory
    output_dir = Path("./data/sampled-dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_small.to_csv(output_dir / "sample-small.csv", index=False)
    sample_large.to_csv(output_dir / "sample-large.csv", index=False)

    return

if __name__ == '__main__':
    main()