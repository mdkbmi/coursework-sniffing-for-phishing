"""
Script to build structured feature dataframes from metadata dataframes.

This script builds a feature dataframe for email classification by processing raw email data.
It extracts various features from email headers and content to identify potential phishing or malicious emails.
The script supports both offline feature extraction (headers, text analysis) and online features that
require external API calls (nameserver lookups, grammar checking).

Features extracted include:
- Header-based: routing paths, authentication results (DMARC, DKIM, SPF), domain matches
- Content-based: text metrics, language analysis, hyperlinks, content types
- Suspicious indicators: hidden text, non-ASCII characters, empty bodies

Usage:
    python build_features_df.py --dataset [dataset_name] --quick [y/n]

Where:
    dataset_name: 'sample-small', 'sample-large', 'train', or 'test'
    quick: 'y' to use cached online features, 'n' to regenerate all features
"""

import pandas as pd
import os
import sys
import click
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.header_features import (
    get_routing_before_ubc, has_dmarc_authentication,  dkim_domain_matches_sender, to_from_match, spf_email_matches_sender,
    check_different_reply_domain, get_internal_server_transfer_count,
    extract_first_server_transfer,
    extract_domain_from_received_header,
    get_name_servers,
    check_sender_name_servers_match,
    get_dkim_result, get_spf_result, get_dmarc_result,
    check_url_present_subject,
)

from src.content_features import (
    word_count, readable_proportion, whitespace_ratio, 
    whitespace_ratio, check_grammar, alphabet_proportion, english_french_proportion, get_content_count, non_ascii_present, hidden_text_present, is_body_empty,
    html_parsing_error, get_hyperlink_proportion
)

from src.data_validation import __validate_features_df

SAMPLED_DATA_DIR = 'data/sampled-dataset/'
FULL_DATA_DIR = 'data/full-dataset/'

SOURCE_DIR = 'raw/'
DESTINATION_DIR = 'processed/'

@click.command()
@click.option('--dataset', type=str, required=True, help="The dataset that you want to build: ['sample-small', 'sample-large', 'train', 'test']")
@click.option('--quick', type=str, required=True, help="If you have the cached file containing previously regenerated features that require online access, do you want to re-generate them? [y/n]")
def main(dataset: str, quick: str) -> None:
    if dataset in ['sample-small', 'sample-large']:
        root_dir = SAMPLED_DATA_DIR
    else:
        root_dir = FULL_DATA_DIR

    source_path = root_dir + SOURCE_DIR + dataset + '.parquet'
    destination_path = root_dir + DESTINATION_DIR + dataset + '.parquet'

    if not os.path.exists(source_path):
        raise FileNotFoundError("Parquet file not found. Did you specify the right dataset? Have you ran scripts/build_original_df.py?")
    
    if not isinstance(quick, str):
        raise TypeError(f"Expected [y/n] in str for --quick, got {type(quick)} instead")
    
    if quick not in ['y', 'n']:
        raise ValueError(f"Expected [y/n] in str for --quick, got {quick} instead")
    
    original_df = pd.read_parquet(source_path)

    routing_before_ubc = get_routing_before_ubc(original_df['Received'])
    first_meaningful_received_header = extract_first_server_transfer(original_df['Received'])
    first_meaningful_received_header_domain = extract_domain_from_received_header(first_meaningful_received_header)
    content_count = pd.DataFrame.from_records(
        get_content_count(original_df['Content_types']), index=original_df.index
    )

    if quick == "y":
        try:
            print('Features requiring online access (including Language Tool server) will not be regenerated. Reading cached file...')
            features_df_online = pd.read_parquet(root_dir + DESTINATION_DIR + dataset + '-partial.parquet')
            assert features_df_online.index.equals(original_df.index), "The indices of features_df_online and original_df must match"
            assert set(features_df_online.columns) == {"name_server_match", "grammar_error_rate", "english_french_proportion"}, "features_df_online contains unexpected columns"
            print("Cached file successfully loaded!")
        except:
            raise FileNotFoundError("`features_df_online` not found.")
    else:
        # Define functions to run in parallel
        def get_name_servers_received():
            return get_name_servers(first_meaningful_received_header_domain)
            
        def get_name_servers_sender():
            return get_name_servers(original_df['From_email_domain'])
            
        def get_lang_proportion():
            return english_french_proportion(original_df['text_plain'])
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            name_servers_received_future = executor.submit(get_name_servers_received)
            name_servers_sender_future = executor.submit(get_name_servers_sender)
            lang_proportion_future = executor.submit(get_lang_proportion)
            
            name_servers_first_meaningful_received_header_domain = name_servers_received_future.result()
            name_servers_sender_domain = name_servers_sender_future.result()
            lang_proportion = lang_proportion_future.result()
    
        features_df_online = pd.DataFrame({
            "name_server_match": check_sender_name_servers_match(
                name_servers_first_meaningful_received_header_domain,
                name_servers_sender_domain
            ),

            "grammar_error_rate": check_grammar(original_df['text_plain'], original_df['Content-Language']),
            "english_french_proportion": lang_proportion,
        })

        destination_path_partial = root_dir + DESTINATION_DIR + dataset + '-partial.parquet'
        os.makedirs(os.path.dirname(destination_path_partial), exist_ok=True)
        features_df_online.to_parquet(destination_path_partial)
        print('Saved features_df_online!')

    features_df_offline = pd.DataFrame({
        "subject": original_df['Subject'].fillna(""),
        "url_present_in_subject": check_url_present_subject(original_df['Subject']),
        "routing_length_before_ubc": routing_before_ubc.apply(len),  # type: ignore
        "dmarc_authentication_present": has_dmarc_authentication(original_df['Authentication-Results']),
        "dkim_sender_domains_match": dkim_domain_matches_sender(
            original_df['DKIM-Signature'],
            original_df['From_email_domain']
        ),
        "to_from_addresses_match": to_from_match(original_df['From_email'], original_df['To_email']),
        "sender_email_spf_match": spf_email_matches_sender(
            original_df['Received-SPF'], original_df['From_email']
        ),
        "different_reply_domains": check_different_reply_domain(
            original_df['From_email_domain'], original_df['Reply-To_domain']
        ),
        "internal_server_transfer_count": get_internal_server_transfer_count(routing_before_ubc),
        # "name_server_match": check_sender_name_servers_match(
        #     name_servers_first_meaningful_received_header_domain,
        #     name_servers_sender_domain
        # ),
        "dkim_result": get_dkim_result(original_df['Authentication-Results']),
        "spf_result": get_spf_result(original_df['Received-SPF']),
        "dmarc_result": get_dmarc_result(original_df['Authentication-Results']),

        "text_clean": original_df['text_clean'],
        "word_count": word_count(original_df['text_clean']),
        "readable_proportion": readable_proportion(original_df['text_clean'], original_df['text_html']),
        "whitespace_ratio": whitespace_ratio(original_df['text_plain']),
        "alphabet_proportion": alphabet_proportion(original_df['text_clean']),
        # "grammar_error_rate": check_grammar(original_df['text_plain'], original_df['Content-Language']),
        # "english_french_proportion": lang_proportion,
        "text_content_count": content_count['text'],
        "multimedia_content_count": content_count['multimedia'],
        "others_content_count": content_count['others'],
        "hyperlink_proportion": get_hyperlink_proportion(
            original_df['text_hyperlinks'],
            word_count(original_df['text_clean'])
        ),

        "non_ascii_present": non_ascii_present(original_df['text_clean']),
        "hidden_text_present": hidden_text_present(original_df['text_html']),
        "empty_body": is_body_empty(
            original_df['text_clean'],
            content_count['multimedia'],
            content_count['others'],
            original_df['text_hyperlinks']
        ),
        "html_parsing_error": html_parsing_error(original_df['text_html']),

        "target_1": original_df['target_1'],
        "target_2": original_df['target_2'],
        "target_3": original_df['target_3'],
    })

    features_df = features_df_online.join(features_df_offline)
    features_df.loc[features_df['empty_body'] == True, 'target_1'] = 'malicious'

    try:
        __validate_features_df(features_df)
    except:
        raise ValueError(f"Errors occurred during validation of features_df")
    
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    features_df.to_parquet(destination_path)

    print("Successfully generated features_df!")

    return

if __name__ == '__main__':
    main()
