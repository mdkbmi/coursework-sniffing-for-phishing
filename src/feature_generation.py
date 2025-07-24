import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_validation import EmailMetadata

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
    check_grammar, alphabet_proportion, english_french_proportion, get_content_count, 
    non_ascii_present, hidden_text_present, is_body_empty, html_parsing_error,
    get_hyperlink_proportion
)

from src.data_validation import EmailMetadata, EmailFeatures

def generate_features_from_metadata(metadata: dict) -> dict:
    """
    Generate and compute features from email metadata for further analysis or classification.

    This function validates the input metadata against an EmailMetadata model,
    processes various aspects of the email (headers, content, authentication),
    and returns a structured dictionary of features.

    Parameters
    ----------
    metadata : dict
        A dictionary containing email metadata including headers (Received, 
        Authentication-Results, DKIM-Signature, etc.), email addresses, 
        content types and body text in various formats.

    Returns
    -------
    dict
        A dictionary containing extracted and computed features including:
        - Authentication features (DMARC, DKIM, SPF results)
        - Routing information (server transfers, internal routing)
        - Content analysis (word count, text proportions, language features)
        - Security indicators (hidden text, non-ASCII characters)
        - Domain relationship features (name server matches, reply domain differences)

    Raises
    ------
    ValueError
        If the input metadata fails validation against EmailMetadata model
        or if the computed features fail validation against EmailFeatures model.

    Notes
    -----
    The function relies on several helper functions to analyze specific aspects
    of the email metadata. The returned features dictionary is validated against
    an EmailFeatures model before being returned.
    """

    try:
        EmailMetadata.model_validate(metadata)
    except Exception as e:
        raise ValueError(f"Validation for metadata failed: {e}")    

    features = {}

    routing_before_ubc = get_routing_before_ubc(metadata['Received'])
    first_meaningful_received_header = extract_first_server_transfer(metadata['Received'])
    first_meaningful_received_header_domain = extract_domain_from_received_header(first_meaningful_received_header)
    name_servers_first_meaningful_received_header_domain = get_name_servers(first_meaningful_received_header_domain)
    name_servers_sender_domain = get_name_servers(metadata['From_email_domain'])
    content_count = get_content_count(metadata['Content_types'])

    features["subject"] = metadata["Subject"]
    features["url_present_in_subject"] = check_url_present_subject(metadata["Subject"])
    features["routing_length_before_ubc"] = len(get_routing_before_ubc(metadata["Received"]))
    features["dmarc_authentication_present"] = has_dmarc_authentication(metadata["Authentication-Results"])
    features["dkim_sender_domains_match"] = dkim_domain_matches_sender(metadata['DKIM-Signature'], metadata['From_email_domain'])
    features["to_from_addresses_match"] = to_from_match(metadata['From_email'], metadata['To_email'])
    features["sender_email_spf_match"] = spf_email_matches_sender(metadata['Received-SPF'], metadata['From_email'])
    features["different_reply_domains"] = check_different_reply_domain(metadata['From_email_domain'], metadata['Reply-To_domain']
    )
    features["internal_server_transfer_count"] = get_internal_server_transfer_count(routing_before_ubc)
    features["name_server_match"] = check_sender_name_servers_match(
        name_servers_first_meaningful_received_header_domain,
        name_servers_sender_domain
    )
    features["dkim_result"] = get_dkim_result(metadata['Authentication-Results'])
    features["spf_result"] = get_spf_result(metadata['Received-SPF'])
    features["dmarc_result"] = get_dmarc_result(metadata['Authentication-Results'])

    features["text_clean"] = metadata['text_clean']
    features["word_count"] = word_count(metadata['text_clean'])
    features["readable_proportion"] = readable_proportion(metadata['text_clean'], metadata['text_html'])
    features["whitespace_ratio"] = whitespace_ratio(metadata['text_plain'])
    features["alphabet_proportion"] = alphabet_proportion(metadata['text_clean'])
    features["grammar_error_rate"] = check_grammar(metadata['text_plain'], metadata['Content-Language'])
    features["english_french_proportion"] = english_french_proportion(metadata['text_plain'])
    features["text_content_count"] = content_count['text']
    features["multimedia_content_count"] = content_count['multimedia']
    features["others_content_count"] = content_count['others']
    features["hyperlink_proportion"] = get_hyperlink_proportion(metadata['text_hyperlinks'], features["word_count"])

    features["non_ascii_present"] = non_ascii_present(metadata['text_clean'])
    features["hidden_text_present"] = hidden_text_present(metadata['text_html'])
    features["empty_body"] = is_body_empty(
        metadata['text_clean'],
        content_count['multimedia'],
        content_count['others'],
        metadata['text_hyperlinks']
    )
    features["html_parsing_error"] = html_parsing_error(metadata['text_html'])
    
    try:
        EmailFeatures.model_validate(features)
        return features
    except Exception as e:
        raise ValueError(f"Validation for features failed: {e}")
