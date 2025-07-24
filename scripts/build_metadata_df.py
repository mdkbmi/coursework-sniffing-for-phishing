"""
Script to build structured metadata dataframes from email datasets.

This script processes email datasets by extracting metadata features from email files,
adding categorical target labels, validating the structure, and saving the results
as Parquet files. It supports both sampled datasets ('sample-small', 'sample-large')
and full datasets ('train', 'test').

The process includes:
1. Reading a CSV file containing email paths and their classification targets
2. Validating the input data structure
3. Extracting metadata features from each email file
4. Joining metadata with target labels
5. Converting target columns to categorical data types
6. Validating the final dataframe
7. Saving the processed dataframe as a Parquet file

Usage:
    python build_metadata_df.py --dataset [dataset_name]
    
Where:
    dataset_name: 'sample-small', 'sample-large', 'train', or 'test'
"""

import pandas as pd
import pandera as pa
import os
import sys
import click

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.metadata_extraction import extract_email_metadata
from src.data_validation import __validate_metadata_df

TARGET_LEVELS = ['target_1', 'target_2']
TARGET_1_LEVELS = ['benign', 'malicious']
TARGET_2_LEVELS = ['ceo_fraud', 'phishing', 'reply-chain-attack',           # malicious
                   'legitimate', 'spam']                                    # benign
TARGET_3_LEVELS = [
    'gift_cards', 'payroll_update', 'wire_transfers', 'third_party', 'outbound',
    'ubc', 'ubc_outbound', 'spearphishing', 'reply-chain-attack',
    'legitimate_email_confirmed', 'spam_false_positive', 'inbound'
]

SAMPLED_DATA_DIR = 'data/sampled-dataset/'
FULL_DATA_DIR = 'data/full-dataset/'

@click.command()
@click.option('--dataset', type=str, required=True, help="The dataset that you want to build: ['sample-small', 'sample-large', 'train', 'test']")
def main(dataset: str) -> None:
    """
    Process and transform email dataset CSV into a structured Parquet file.
    This function reads a CSV file containing email paths, extracts metadata from the emails,
    validates the data structure, and saves the processed dataframe as a Parquet file.

    Parameters
    ----------
    dataset : str
        Name of the dataset to process. Valid options are 
        ['sample-small', 'sample-large', 'train', 'test'].

    Returns
    -------
    None
        Function saves the processed dataframe to a Parquet file but does not return any value.

    Raises
    ------
    FileNotFoundError
        If the source CSV file does not exist.
    ValueError
        If the email list file does not match the expected schema, or if validation 
        of the final dataframe fails.
    """
    if dataset in ['sample-small', 'sample-large']:
        source_path = SAMPLED_DATA_DIR + dataset + '.csv'
        destination_path = SAMPLED_DATA_DIR + '/raw/' + dataset + '.parquet'
    else:
        source_path = FULL_DATA_DIR + dataset + '.csv'
        destination_path = FULL_DATA_DIR + '/raw/' + dataset + '.parquet'

    if not os.path.exists(source_path):
        raise FileNotFoundError("CSV file consisting of email paths not found. Have you done `make prepare-dataset`?")
    
    email_list = pd.read_csv(
        source_path,
        index_col=0
    )

    schema_email_list = pa.DataFrameSchema(
        {
            "target_1": pa.Column(str),
            "target_2": pa.Column(str),
            "target_3": pa.Column(str),
        }
    )

    try:
        schema_email_list.validate(email_list)
    except:
        raise ValueError(f"{source_path} is not the expected file")

    email_metadata = pd.DataFrame.from_records(
        extract_email_metadata(email_list.index.to_list())
    ).set_index('path')

    metadata_df = email_metadata.join(email_list)
    metadata_df = metadata_df[metadata_df['target_3'] != 'self_phishing']
    metadata_df['target_1'] = pd.Categorical(
        metadata_df['target_1'], categories=TARGET_1_LEVELS, ordered=True
    )
    metadata_df['target_2'] = pd.Categorical(
        metadata_df['target_2'], categories=TARGET_2_LEVELS, ordered=False
    )
    metadata_df['target_3'] = pd.Categorical(
        metadata_df['target_3'], categories=TARGET_3_LEVELS, ordered=False
    )

    try:
        __validate_metadata_df(metadata_df)
    except:
        raise ValueError(f"Errors occurred during validation of metadata_df")
    
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    metadata_df.to_parquet(destination_path)
    print("Built metadata_df successfully!")

    return

if __name__ == '__main__':
    main()
