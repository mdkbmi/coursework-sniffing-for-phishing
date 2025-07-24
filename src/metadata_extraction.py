import warnings
import email
from email.policy import default
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Tuple, List, Union, Optional

import pandas as pd
from bs4 import BeautifulSoup

from src.content_features import extract_hyperlinks
from src.data_validation import EmailMetadata

def get_name_from_email_header(header_field: str | pd.Series) -> str | pd.Series | None:
    """
    Extract display name from an email field's 'From' or similar headers.
        
    This function parses email header fields to extract the human-readable name
    part, which helps identify how the sender presented themselves in the email.
        
    Parameters
    ----------
    header_field : str or pandas.Series
        The email header field content (like 'From', 'To', 'Reply-To', etc.)
        that typically contains name and email address
            
    Returns
    -------
    str, pandas.Series or None
        The extracted name part of the email header
        - If input is a string: Returns name as string or header_field if no name found
        - If input is a Series: Returns a Series of names
        - Returns None if the input is None
            
    Examples
    --------
    >>> get_name_from_email_header('John Doe <john@example.com>')
    'John Doe'
    >>> get_name_from_email_header('no-name@example.com')
    'no-name@example.com'
    """
    if isinstance(header_field, pd.Series):
        return header_field.apply(get_name_from_email_header) # type: ignore

    if not header_field:
        warnings.warn("None entry found in `header_field`")
        return None
    
    if not isinstance(header_field, str):
        raise ValueError("`header_field` must be str or pandas.Series")
    
    name, email = parseaddr(header_field)
    
    if (name == '' and email == ''):
        name = header_field

    return name

def get_email_addr_from_email_header(header_field: str | pd.Series) -> str | pd.Series | None:
    """
    Extract email address from an email field's 'From' or similar headers.
        
    This function parses email header fields to extract the email address part,
    which helps identify the actual address used by the sender.
        
    Parameters
    ----------
    header_field : str or pandas.Series
        The email header field content (like 'From', 'To', 'Reply-To', etc.)
        that typically contains name and email address
            
    Returns
    -------
    str, pandas.Series or None
        The extracted email address part of the email header
        - If input is a string: Returns email address as string or None if no valid address found
        - If input is a Series: Returns a Series of email addresses
        - Returns None if the input is None
            
    Examples
    --------
    >>> get_email_addr_from_email_header('John Doe <john@example.com>')
    'john@example.com'
    >>> get_email_addr_from_email_header('invalid-format')
    None
    """
    if isinstance(header_field, pd.Series):
        return header_field.apply(get_email_addr_from_email_header) # type: ignore

    if not header_field:
        warnings.warn("None entry found in `header_field`")
        return None
    
    if not isinstance(header_field, str):
        raise ValueError("`header_field` must be str or pandas.Series")
    
    name, email = parseaddr(header_field)

    if email == '':
        return None
    return email

def get_email_domain(email_address: str | pd.Series | None) -> str | pd.Series | None:
    """
    Get domain of an email address.
        
    Parameters
    ----------
    email_address : str or pandas.Series
        Email address or series of email addresses
        
    Returns
    -------
    str, pandas.Series or None
        The extracted domain part of the email address(es)
        - If input is a string: Returns domain as string or None if not a valid email
        - If input is a Series: Returns a Series of domains
        
    Examples
    --------
    >>> get_email_domain('user@example.com')
    'example.com'
    >>> get_email_domain('invalid-email')
    None
    """
    if isinstance(email_address, pd.Series):
        return email_address.apply(get_email_domain) # type: ignore

    if not email_address:
        warnings.warn("None entry found in `email_address`")
        return None
    
    if not isinstance(email_address, str):
        raise ValueError("`email_address` must be str or pandas.Series or None")

    parts = email_address.strip().split('@')
    
    if len(parts) == 2 and parts[1]:
        return parts[1].lower()
    else:
        return None

def extract_body(msg: EmailMessage) -> Tuple[List[str], str, str, Optional[str]]:
    """
    Recursively extracts MIME content types, plain text, and HTML from an EmailMessage object.

    This function performs a depth-first traversal of the email structure to collect:
    
    - All MIME content types found in the message parts.
    - The first encountered plain text body (`text/plain`), cleaned of excess whitespace and UBC's caution tag.
    - The first encountered HTML body (`text/html`).
    - A fallback plain text version extracted from HTML if no `text/plain` is found.

    Parameters
    ----------
    msg : EmailMessage
        The email message object from which to extract content.

    Returns
    -------
    tuple :
        A tuple containing:
        - parts (list of str): All MIME content types found (e.g., 'text/plain', 'text/html', etc.).
        - text_plain (str): The raw plain text body of the email, or extracted from HTML if not available.
        - text_clean (str): Cleaned version of `text_plain`, with whitespace trimmed and caution tags removed.
        - text_html (str): The raw HTML body of the email, if present; otherwise None.

    Raises
    ------
    TypeError
        If the input `msg` is not an instance of `EmailMessage`.

    Notes
    -----
    - Only the first encountered plain and HTML bodies are extracted.
    - If a `text/plain` part is not available, plain text is extracted from the HTML using BeautifulSoup.
    - The cleaning step removes excess whitespace and UBC’s caution tag: `[CAUTION: Non-UBC Email]`.
    """
    
    if not isinstance(msg, EmailMessage):
        raise TypeError(f"Expect msg to be a EmailMessage but got {type(msg)}")
        
    parts = list()
    text_plain = str() 
    text_plain_from_html = str() # use this if 'text_plain' is blank 
    text_clean = str()
    text_html = None 

    content_type = msg.get_content_type()
    parts.append(content_type)
    
    if msg.is_multipart():
        for part in msg.iter_parts():
            sub_parts, sub_text_plain, sub_text_clean, sub_text_html = extract_body(part)
            parts.extend(sub_parts)

            # Prioritize first plain or html content found
            if text_plain == '' and sub_text_plain != '':
                text_plain = sub_text_plain
            if text_html is None and sub_text_html is not None: 
                text_html = sub_text_html
    else:
        disposition = msg.get_content_disposition()

        if content_type == 'text/html':
            try:
                text_html = msg.get_content()
            except Exception:
                pass
            try:
                text_plain_from_html = BeautifulSoup(text_html, 'html.parser').get_text()
            except Exception:
                pass

        if content_type == 'text/plain':
            text_plain = msg.get_content()

    parts = list(parts) if parts else list()

    if text_plain != '': # prioritize plain text from text/plain over plain text parsed from html 
        text_clean = " ".join(text_plain.split()).replace('[CAUTION: Non-UBC Email]', '').lstrip()

        return parts, text_plain, text_clean, text_html
    
    else:
        text_clean = " ".join(text_plain_from_html.split()).replace('[CAUTION: Non-UBC Email]', '').lstrip()

    return parts, str(text_plain_from_html), str(text_clean), text_html

def __extract_email_metadata(path: str) -> dict:
    """
    Extract metadata from an email file.

    Parameters
    ----------
    path : str
        Path to the email file to be processed.

    Returns
    -------
    dict
        Dictionary containing email metadata with the following keys:
        - 'From_email': Extracted sender email address
        - 'From_email_domain': Domain of the sender email
        - 'To_email': Extracted recipient email address
        - 'To_email_domain': Domain of the recipient email
        - 'Subject': Email subject line
        - 'Received': All 'Received' headers
        - 'Authentication-Results': Authentication results header
        - 'Received-SPF': SPF verification result
        - 'DKIM-Signature': DKIM signature if available
        - 'Reply-To_domain': Domain of the Reply-To address if available
        - 'Content-Language': Content language if specified
        - 'Content_types': List of content types in the email
        - 'text_plain': Plain text version of the email body
        - 'text_clean': Cleaned text version of the email body
        - 'text_html': HTML version of the email body
        - 'text_hyperlinks': Extracted hyperlinks from HTML content

    Raises
    ------
    ValueError
        If path is None, if email file cannot be opened/parsed, or if metadata validation fails.
    TypeError
        If the parsed object is not an EmailMessage.

    Notes
    -----
    This function uses the email module to parse the email file and extract
    various metadata fields. The returned metadata is validated against the
    EmailMetadata model.
    """

    if path is None:
        raise ValueError("path must be provided.")
        
    try:
        with open(path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=default)
    except Exception as e:
        raise ValueError(f"Failed to open or parse email file {path}: {e}")
    
    if not isinstance(msg, EmailMessage):
        raise TypeError(f"Expected EmailMessage object, but got {type(msg).__name__}")
        
    metadata = {}

    if msg['From']:
        metadata['From_email'] = get_email_addr_from_email_header(msg['From'])
        metadata['From_email_domain'] = get_email_domain(metadata['From_email'])
    else: 
        metadata['From_email'] = None
        metadata['From_email_domain'] = None

    if msg['To']:
        metadata['To_email'] = get_email_addr_from_email_header(msg['To'])
        metadata['To_email_domain'] = get_email_domain(metadata['To_email'])
    else: 
        metadata['To_email'] = None
        metadata['To_email_domain'] = None

    metadata['Subject'] = str(msg['Subject']) if msg['Subject'] else str()
    metadata['Received'] = msg.get_all('Received')
    metadata['Authentication-Results'] = str(msg['Authentication-Results'])
    metadata['Received-SPF'] = str(msg['received-spf']) if msg['received-spf'] else None
    metadata['DKIM-Signature'] = str(msg['DKIM-Signature']) if msg['DKIM-Signature'] else None
    metadata['Reply-To_domain'] = str(get_email_domain(
        get_email_addr_from_email_header(msg['Reply-To'])
    )) if msg['Reply-To'] else None
    metadata['Content-Language'] = msg['Content-Language'] if msg['Content-Language'] else None
    metadata['Content_types'], metadata['text_plain'], metadata['text_clean'], metadata['text_html'] = extract_body(msg)
    metadata['text_hyperlinks'] = extract_hyperlinks(metadata['text_html']) # type: ignore

    try:
        EmailMetadata.model_validate(metadata)
        return metadata
    except Exception as e:
        raise ValueError(f"Validation for metadata failed: {e}")


def extract_email_metadata(path: str | list) -> dict | list:
    """
    Extract metadata from email file(s).

    This function parses email file(s) and extracts various metadata including sender/receiver 
    information, email headers, and content.

    Parameters
    ----------
    path : str or list
        Path to a single email file (str) or a list of paths to multiple email files.
    Returns
    -------
    dict or list
        If input is a single path (str): Returns a dictionary containing the extracted metadata.
        If input is a list of paths: Returns a list of dictionaries, each containing metadata 
        for one email.
    Raises
    ------
    TypeError
        If the input path is neither a string nor a list.
    Notes
    -----
    Extracted metadata includes:
    - Basic information (path, multipart status)
    - Sender information (name, email, domain)
    - Receiver information (name, email, domain)
    - Email headers (Subject, Received, Authentication-Results, etc.)
    - Content information (plain text, HTML, hyperlinks, attachment types)
    """
    
    if isinstance(path, str):
        return __extract_email_metadata(path)
    

    if isinstance(path, list):
        extracted_dict_list = []

        for individual_path in path:
            extracted_dict = {'path': individual_path}
            extracted_dict.update(__extract_email_metadata(individual_path))
            extracted_dict_list.append(extracted_dict)
            
        return extracted_dict_list
    
    else:
        raise TypeError(f"Expect path to be either a str or list but got {type(path)}")
