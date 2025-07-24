# extract_text_features.py
# author: Danish Karlin Isa
# created: 2025-05-15

from __future__ import annotations
from typing import Union
import pandas as pd
import numpy as np
from lxml import html
from lxml.html import HTMLParser
from lxml import etree
import re
import requests
from lingua import LanguageDetector, LanguageDetectorBuilder
import time
from array import array
import warnings

def __non_ascii_present(text_clean: str) -> bool:
    """Private method for `is_non_ascii_present`.

    Parameters
    ----------
    text : str
        Some text to check for non-ASCII characters.
        
    Returns
    -------
    bool
        Returns True if non-ASCII characters are present, False otherwise.
    """
    if not isinstance(text_clean, str) or pd.isna(text_clean):
        return False
    
    return not text_clean.isascii()

def non_ascii_present(text_clean: Union[str, pd.Series[str]]) -> Union[bool, pd.Series[bool]]:
    """Check if non-ASCII characters are present in text.
    
    Parameters
    ----------
    text : str or pandas.Series
        Some text or a series of texts to check for non-ASCII characters.
        
    Returns
    -------
    bool or pandas.Series
        Returns True if non-ASCII characters are present, False otherwise.
        If input is a pandas Series, returns a Series of boolean values.
        
    Example
    -------
    >>> text = "Hello world!"
    >>> is_non_ascii_present(text)
    False
    >>> text = "Hello world! 你好"
    >>> is_non_ascii_present(text)
    True
    >>> series = pd.Series(["ASCII only", "Non-ASCII: 你好"])
    >>> is_non_ascii_present(series)
    0    False
    1     True
    dtype: bool
    """
    if isinstance(text_clean, pd.Series):
        return text_clean.apply(__non_ascii_present)
    
    else:
        return __non_ascii_present(text_clean)
    


def __hidden_text_present(text_html: Union[str, bytes]) -> bool:
    """Private method for `hidden_text_present`.

    Parameters
    ----------
    text_html : str or pandas.Series
        HTML content to check for hidden text, either a single HTML string or
        a pandas Series containing HTML content.
        
    Returns
    -------
    bool or pandas.Series
        Returns True if hidden text is found, False otherwise.
        If input is a pandas Series, returns a Series of booleans.
    """
    if pd.isna(text_html) or text_html is None:
        return False
        
    try:
        if isinstance(text_html, bytes):
            text_html = text_html.decode('utf-8', errors='replace')
            
        hidden_patterns = [
            'style="color: white;"',
            'style="color:#ffffff"',
            'style="display:none"',
            'style="visibility:hidden"',
            'style="opacity:0"',
            'style="font-size:0px"',
            'style="height:0px"',
            '<div hidden',
            'class="hidden"'
        ]
        
        for pattern in hidden_patterns:
            if pattern in str(text_html):
                return True
                
        return False
    
    except:
        return False

def hidden_text_present(text_html: Union[str, bytes, pd.Series[Union[str, bytes]]]) -> Union[bool, pd.Series[bool]]:
    """Check if hidden text is present in HTML content.
    
    Parameters
    ----------
    text_html : str or pandas.Series
        HTML content to check for hidden text, either a single HTML string or
        a pandas Series containing HTML content.
        
    Returns
    -------
    bool or pandas.Series
        Returns True if hidden text is found, False otherwise.
        If input is a pandas Series, returns a Series of booleans.
        
    Example
    -------
    >>> html = '<p style="color: white;">Hidden text</p>'
    >>> is_hidden_text_present(html)
    True
    >>> html = '<p>Visible text</p>'
    >>> is_hidden_text_present(html)
    False
    >>> df = pd.DataFrame({'html': ['<p>Visible</p>', '<p style="color: white;">Hidden</p>']})
    >>> is_hidden_text_present(df.html)
    0    False
    1     True
    dtype: bool
    """  
    if isinstance(text_html, pd.Series):
        return text_html.apply(__hidden_text_present)
    else:
        return __hidden_text_present(text_html)
    


def __html_parsing_error(text_html: Union[str, bytes]) -> int:
    """Private method for `html_parsing_error`

    Parameters
    ----------
    text_html : str
        HTML content to check for parsing errors in a single HTML string.
        
    Returns
    -------
    int
        Returns 1 if parsing errors are found, 0 if no errors are found, 
        and -1 if an exception occurred during parsing.
        If input is a pandas Series, returns a Series of int values.
    """
    if pd.isna(text_html) or text_html is None:
        return 0
    
    try:
        if isinstance(text_html, bytes):
            text_html = text_html.decode('utf-8', errors='replace')
                
        parser = HTMLParser(recover=True)
        document = etree.fromstring(text_html, parser)
        return 1 if parser.error_log else 0
    
    except:
        return -1

def html_parsing_error(text_html: Union[str, bytes, pd.Series]) -> Union[int, pd.Series]:
    """Check if there are errors when parsing HTML.
    
    Parameters
    ----------
    text_html : str or pandas.Series
        HTML content to check for parsing errors, either a single HTML string or
        a pandas Series containing HTML content.
        
    Returns
    -------
    int or pandas.Series
        Returns 1 if parsing errors are found, 0 if no errors are found, 
        and -1 if an exception occurred during parsing.
        If input is a pandas Series, returns a Series of int values.
        
    Example
    -------
    >>> html = '<div>Valid HTML</div>'
    >>> check_error_while_parsing(html)
    0
    >>> html_with_error = '<div>Invalid HTML<div'
    >>> check_error_while_parsing(html_with_error)
    1
    >>> df = pd.DataFrame({'html': ['<div>Valid</div>', '<div>Invalid<div>']})
    >>> check_error_while_parsing(df.html)
    0    0
    1    1
    dtype: int64
    """
    if isinstance(text_html, pd.Series):
        results = [__html_parsing_error(t) for t in text_html]
                
        return pd.Series(results, index=text_html.index)
    else:
        return __html_parsing_error(text_html)



def __word_count(text_clean: str) -> int:
    """Private method for `word_count`.

    Parameters
    ----------
    text_clean : str
        Some text that a word count is required from.
        Text must be free of all formatting elements.

    Returns
    -------
    int
        Word count of the given text.
    """
    return len(text_clean.split())

def word_count(text_clean: Union[str, pd.Series[str]]) -> Union[int, pd.Series[int]]:
    """Obtain word count of a given text or series of texts.

    Parameters
    ----------
    text_clean : str or pandas.Series
        Some text or a series of texts that a word count is required from.
        Text must be free of all formatting elements.

    Returns
    -------
    int or pandas.Series
        Word count of the given text. If input is a pandas Series,
        returns a Series of word counts.

    Example
    -------
    >>> text = "Hello world!"
    >>> get_word_count(text)
    2
    >>> series = pd.Series(["Hello world!", "This is a test."])
    >>> get_word_count(series)
    0    2
    1    4
    dtype: int64
    """
    if isinstance(text_clean, pd.Series):
        return text_clean.apply(__word_count)
    else:
        return __word_count(text_clean)
    


def __readable_proportion(text_clean: str, text_html: Union[str, bytes]) -> float:
    """Private method for `readable_proportion`

    Parameters
    ----------
    text_clean : str
        Clean text extracted from the HTML content.
    text_html : str or bytes
        Original HTML content.
        
    Returns
    -------
    float
        Proportion of readable text calculated as the ratio of clean text length
        to HTML content length.
    """
    if pd.isna(text_html) or not text_html:
        return 0.0
    
    else:
        try:
            prop = len(text_clean) / len(text_html)

            if not (prop >= 0 and prop <= 1):
                prop = 0.0

            return prop
        except (TypeError, ZeroDivisionError):
            return 0.0

def readable_proportion(text_clean: Union[str, pd.Series[str]], text_html: Union[str, bytes, pd.Series[Union[str, bytes]]]) -> Union[float, pd.Series[float]]:
    """Calculates the ratio between the length of the extracted clean text and
    the original HTML content length. This provides an indicator of how much
    of the original HTML is actual human-readable content.
    
    Parameters
    ----------
    text_clean : str or pandas.Series
        Clean text extracted from the HTML content.
    text_html : str or bytes or pandas.Series
        Original HTML content.
        
    Returns
    -------
    float or pandas.Series
        Proportion of readable text calculated as the ratio of clean text length
        to HTML content length. Returns a single float for string inputs, or a
        Series of proportions if input is a pandas Series.
        
    Notes
    -----
    The proportion is calculated as:
    len(text_clean) / len(text_html)
        
    Example
    -------
    >>> html = "<html><body>Hello world!</body></html>"
    >>> clean = "Hello world!"
    >>> get_prop_readable_text(clean, html)
    0.375
    >>> df = pd.DataFrame({'clean': ['Hello', 'Test'], 'html': ['<p>Hello</p>', '<div>Test</div>']})
    >>> get_prop_readable_text(df.clean, df.html)
    0    0.454545
    1    0.333333
    dtype: float64

    Raises
    ------
    ValueError
        If input pandas Series don't have the same index.
    TypeError
        If inputs are not strings or pandas Series.
    """
    if isinstance(text_clean, pd.Series) and isinstance(text_html, pd.Series):
        if not text_clean.index.equals(text_html.index):
            raise ValueError("Series must have the same index")
        
        result = pd.Series(index=text_clean.index, dtype=float)

        for i in text_clean.index:
            result[i] = __readable_proportion(text_clean[i], text_html[i])

        return result
    
    elif isinstance(text_clean, str) and (isinstance(text_html, (str, bytes)) or not text_html):       # type: ignore
        return __readable_proportion(text_clean, text_html) # type: ignore # type: ignore
    
    else:
        raise TypeError("Both `text_clean` and `text_html` should either be strings or pandas.Series")
    


def __whitespace_ratio(text_plain: str) -> float:
    """Private method for `whitespace_ratio`.

    Parameters
    ----------
    text : str
        A string to analyze for whitespace characters.
        
    Returns
    -------
    float
        Ratio of whitespace characters to total characters.
    """
    if pd.isna(text_plain) or not isinstance(text_plain, str) or len(text_plain) == 0:
        return 0.0
            
    whitespace_pattern = r'\s'
    whitespace_matches = re.findall(whitespace_pattern, text_plain)
        
    return len(whitespace_matches) / len(text_plain)

def whitespace_ratio(text_plain: Union[str, pd.Series[str]]) -> Union[float, pd.Series[float]]:
    """Calculate the ratio of whitespace characters to total characters in text.
    
    Parameters
    ----------
    text_plain : str or pandas.Series
        A string or series of strings to analyze for whitespace characters.
        
    Returns
    -------
    float or pandas.Series
        Ratio of whitespace characters to total characters. If input is a pandas Series,
        returns a Series of ratios.
        
    Formula
    -------
    Whitespace ratio = (Number of whitespace characters) / (Total number of characters)
        
    Example
    -------
    >>> text = "Hello world!"
    >>> whitespace_ratio(text)
    0.0833  # 1 space ÷ 12 total characters
    >>> series = pd.Series(["Hello world!", "Text with\ttab and\nnewline"])
    >>> whitespace_ratio(series)
    0    0.083333
    1    0.115385
    dtype: float64
    """
    if isinstance(text_plain, pd.Series):
        return text_plain.apply(__whitespace_ratio)
    else:
        return __whitespace_ratio(text_plain)
    


def __alphabet_proportion(text_clean: str) -> float:
    """Private method for `alphabet_proportion`

    Parameters
    ----------
    text_clean : str
        A string to analyze for alphabetical characters.
        
    Returns
    -------
    float
        Proportion of alphabetical characters to total characters. 
    """
    try:
        return sum([c.isalpha() for c in text_clean]) / len(text_clean)
    except ZeroDivisionError:
        return 0.0

def alphabet_proportion(text_clean: Union[str, pd.Series[str]]) -> Union[float, pd.Series[float]]:
    """Calculate the proportion of alphabetical characters in text.

    Parameters
    ----------
    text_clean : str or pandas.Series
        A string or series of strings to analyze for alphabetical characters.
        
    Returns
    -------
    float or pandas.Series
        Proportion of alphabetical characters to total characters. 
        If input is a pandas Series, returns a Series of proportions.
        
    Formula
    -------
    Proportion of alphabetical characters = (Number of alphabetical characters) / (Total number of characters)
        
    Example
    -------
    >>> text = "Hello world! 123"
    >>> get_prop_alphabets(text)
    0.6875  # 11 alphabetical characters ÷ 16 total characters
    >>> series = pd.Series(["Hello world!", "Text with 42 numbers"])
    >>> get_prop_alphabets(series)
    0    0.833333
    1    0.722222
    dtype: float64
    """
    if isinstance(text_clean, pd.Series):
        return text_clean.apply(__alphabet_proportion)
    
    else:
        return __alphabet_proportion(text_clean)
    


def __check_grammar(text_plain: str, language: str, url: str) -> float:
    if language:
        if language.lower() not in ['en', 'en-us', 'en-gb', 'en-ca', 'en-au']:
            return 1.0

    params = {
        "text": text_plain,
        "language": 'en',
        "disabledCategories": 'SEMANTICS',
    }

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(url, data=params, timeout=5)

            if response.status_code == 200:
                result = response.json()

                try:
                    prop = len(result['matches']) / word_count(text_plain)

                    if not (prop >= 0 and prop <= 1):   # type: ignore
                        prop = 1.0

                    return prop                         # type: ignore
                
                except ZeroDivisionError:
                    return 0.0
            else:
                retry_count += 1

                if retry_count >= max_retries:
                    print(f"Warning: LanguageTool server returned status {response.status_code} after {max_retries} attempts - returning 0 grammar errors")

                    return 0.0
                
                print(f"LanguageTool server returned status {response.status_code}, attempt ({retry_count}/{max_retries})")

                continue
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f'Warning: LanguageTool server connection failed after {max_retries} attempts - returning 0 grammar errors')
                break
            print(f"LanguageTool connection attempt {retry_count}/{max_retries} failed: {str(e)}")

            time.sleep(1)
    
    return 0.0


def check_grammar(text_plain: Union[str, pd.Series[str]], language: Union[None, str, pd.Series[str]], url: str="http://localhost:8081/v2/check") -> Union[float, pd.Series[float]]:
    """Checks grammar of a given text. 

    Parameters
    ----------
    text_plain : str or pandas.Series
        Some text or a series of texts.
        Text must be free of all HTML formatting elements.
    language : str, None, or pandas.Series, default None
        The language code of the text (e.g., 'en' for English).
    url : str, default "http://localhost:8081/v2/check"
        The endpoint URL for the Language Tool server.

    Returns
    -------
    float or pandas.Series
        Ratio of grammatical mistakes to word count. If 
        input is a pandas Series, returns a Series of ratios.

    Raises
    ------
    ValueError
        If text_plain and language are Series with different indices.
    TypeError
        If text_plain and language are not both strings or both Series.

    Example
    -------
    >>> text = "This are a example with grammar errors."
    >>> check_grammar(text)
    0.2
    >>> series = pd.Series(["This is correct.", "These is not correct."])
    >>> check_grammar(series)
    0    0.0
    1    0.2
    dtype: float64
    """
    if isinstance(text_plain, pd.Series) and isinstance(language, pd.Series):
        if not text_plain.index.equals(language.index):
            raise ValueError("Series must have the same index")
        
        result = pd.Series(index=text_plain.index, dtype=float)

        for i in result.index:
            result[i] = __check_grammar(text_plain[i], language[i], url)

        return result
    
    elif isinstance(text_plain, str) and (isinstance(language, str) or not language):       # type: ignore
        return __check_grammar(text_plain, language, url) # type: ignore
    
    else:
        raise TypeError("Both `text_plain` and `language` should either be strings or pandas.Series")


def __english_french_proportion(text_plain: str, detector: LanguageDetector, min_word_count: int=5) -> float:
    """Private method for `english_french_proportion`.

    Parameters
    ----------
    text_plain : str
        A string to analyze for English and French content.
    detector : LanguageDetector
        A lingua language detector instance.
    min_word_count : int, default=5
        The minimum word count for a language segment to be considered significant.

    Returns
    -------
    float
        Proportion of text that is identified as either English or French.
        Returns a value between 0 and 1, where 1 means all text is either English or French.
    """
    if not text_plain:
        return 0.0
    
    languages = detector.detect_multiple_languages_of(text_plain)

    list_langs = [
        language.language.iso_code_639_1.name for language in languages \
        if language.word_count > min_word_count
    ]
    list_word_count = [
        language.word_count for language in languages \
        if language.word_count > min_word_count
    ]

    msg_language = pd.DataFrame({
        'language': list_langs,
        'count': list_word_count,
    }).groupby('language').sum().query('`count` > 10')

    if len(msg_language) == 0:
        return 0.0

    msg_language['count'] /= sum(msg_language['count'])

    prop_EN = msg_language['count']['EN'] if 'EN' in msg_language.index else 0
    prop_FR = msg_language['count']['FR'] if 'FR' in msg_language.index else 0

    return prop_EN + prop_FR

def english_french_proportion(text_plain: Union[str, pd.Series[str]]) -> Union[float, pd.Series[float]]:
    """Calculate the proportion of English and French text in the content.

    Parameters
    ----------
    text_plain : str or pandas.Series
        A string or series of strings to analyze for English and French content.

    Returns
    -------
    float or pandas.Series
        Proportion of text that is identified as either English or French.
        Returns a value between 0 and 1, where 1 means all text is either English or French.
        If input is a pandas Series, returns a Series of proportions.

    Example
    -------
    >>> text = "This is English text."
    >>> english_french_proportion(text)
    1.0
    >>> text = "This is English text with some other languages like こんにちは"
    >>> english_french_proportion(text)
    0.8
    >>> series = pd.Series(["English only", "English and français mélangés"])
    >>> english_french_proportion(series)
    0    1.0
    1    1.0
    dtype: float64
    """
    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
    
    if isinstance(text_plain, pd.Series):
        return text_plain.apply(__english_french_proportion, detector=detector)
    
    else:
        return __english_french_proportion(text_plain, detector)

def get_content_count(content_types: list | array | pd.Series) -> dict | pd.Series:
    """
    Count different content types in an email.

    This function analyzes the Content-Type headers from an email and categorizes them
    into text, multimedia, and other types, providing a count for each category.

    Parameters
    ----------
    content_types : list, array.array, or pandas.Series
        A list or Series of content type strings from email headers
        
    Returns
    -------
    dict or pandas.Series
        Dictionary or Series of dictionaries with counts for each content type category:
        - text: Count of text/* content types (text/plain, text/html, etc.)
        - multimedia: Count of image/*, audio/*, video/* content types
        - others: Count of all other content types
        
    Examples
    --------
    >>> get_content_count(['text/plain', 'text/html', 'image/jpeg'])
    {'text': 2, 'multimedia': 1, 'others': 0}
    """
    if isinstance(content_types, pd.Series):
        return content_types.apply(get_content_count)  # type: ignore
    
    if not isinstance(content_types, pd.Series) and hasattr(content_types, 'tolist'):
        content_types = content_types.tolist() # type: ignore
    
    if not isinstance(content_types, list):
        raise ValueError("`content_types` must be a list or array.array")
        
    count = {
        'text': int(0),
        'multimedia': int(0),
        'others': int(0),
    }
    
    for content_type in content_types:
        if not isinstance(content_type, str):
            raise ValueError
        
        if content_type.startswith('multipart/'):
            continue
        elif content_type.startswith('text/'):
            count['text'] += 1
        elif content_type.startswith('image/') or content_type.startswith('audio/') or content_type.startswith('video/'):
            count['multimedia'] += 1
        else:
            count['others'] += 1
    
    return count

def __is_body_empty(text_clean: str, multimedia_count: int, others_count: int, text_hyperlinks: list) -> bool:
    """Private method for `is_body_empty`.
    
    Parameters
    ----------
    text_clean : str
        Clean text extracted from the email body.
    multimedia_count : int
        Count of multimedia attachments.
    others_count : int
        Count of other attachments.
    text_hyperlinks : list
        List of hyperlinks in the text.
        
    Returns
    -------
    bool
        True if the body is considered empty, False otherwise.
    """
    if not isinstance(text_clean, (str, np.str_)):
        raise TypeError(f"text_clean must be a string, got {type(text_clean).__name__}")
    
    if not isinstance(multimedia_count, (int, np.integer)):
        raise TypeError(f"multimedia_count must be an integer, got {type(multimedia_count).__name__}")
    
    if not isinstance(others_count, (int, np.integer)):
        raise TypeError(f"others_count must be an integer, got {type(others_count).__name__}")
        
    if not hasattr(text_hyperlinks, '__len__'):
        raise TypeError(f"text_hyperlinks must be array-like, got {type(text_hyperlinks).__name__}")

    if text_clean == "" and multimedia_count == 0 and others_count == 0 and len(text_hyperlinks) == 0:
        return True
    
    return False

def is_body_empty(text_clean: str | pd.Series, multimedia_count: int | pd.Series, others_count: int | pd.Series, text_hyperlinks: set | pd.Series) -> bool | pd.Series:
    """Check if an email body is empty.
    
    Parameters
    ----------
    text_clean : str or pandas.Series
        Clean text extracted from the email body.
    multimedia_count : int or pandas.Series
        Count of multimedia attachments.
    others_count : int or pandas.Series
        Count of other attachments.
    text_hyperlinks : set or pandas.Series
        Set of hyperlinks in the text.
        
    Returns
    -------
    bool or pandas.Series
        True if the body is considered empty, False otherwise.
        If inputs are pandas Series, returns a Series of boolean values.
        
    Example
    -------
    >>> is_body_empty("", 0, 0, [])
    True
    >>> is_body_empty("Hello", 0, 0, [])
    False
    >>> df = pd.DataFrame({
    ...     'text': ["", "Hello"],
    ...     'multimedia': [0, 1],
    ...     'others': [0, 0],
    ...     'links': [[], ["http://example.com"]]
    ... })
    >>> is_body_empty(df.text, df.multimedia, df.others, df.links)
    0    True
    1    False
    dtype: bool
    """
    if (
        isinstance(text_clean, pd.Series) and isinstance(multimedia_count, pd.Series) and \
        isinstance(others_count, pd.Series) and isinstance(text_hyperlinks, pd.Series)
    ):
        for series in [multimedia_count, others_count, text_hyperlinks]:
            if not isinstance(series, pd.Series):
                raise TypeError("All inputs must be pandas Series when the first argument is a Series")
            if not text_clean.index.equals(series.index):
                raise ValueError("All Series must have the same index")
        
        result = pd.Series(index=text_clean.index, dtype=bool)
        
        for idx in text_clean.index:
            result[idx] = __is_body_empty(
                text_clean[idx],  # type: ignore
                multimedia_count[idx],  # type: ignore
                others_count[idx],  # type: ignore
                text_hyperlinks[idx] # type: ignore
            )
        
        return result
    
    if (
        isinstance(text_clean, str) and isinstance(multimedia_count, int) and \
        isinstance(others_count, int) and isinstance(text_hyperlinks, set)
    ):
        return __is_body_empty(text_clean, multimedia_count, others_count, text_hyperlinks) # type: ignore
    
    raise TypeError("Check input types.")

def extract_hyperlinks(text_html: bytes | str | pd.Series) -> set | pd.Series:
    """
    Extract hyperlinks from HTML content.

    This function parses HTML content to extract URL links (<a href> tags),
    which helps analyze email content for phishing indicators and malicious URLs.

    Parameters
    ----------
    text_html : bytes, str, or pandas.Series
        HTML content from an email or collection of emails

    Returns
    -------
    set or pandas.Series
        Set of extracted hyperlinks or Series of sets for multiple emails
        - If input is a string or bytes: Returns a set of URLs
        - If input is a Series: Returns a Series of URL sets

    Examples
    --------
    >>> extract_hyperlinks('<a href="https://example.com">Link</a>')
    {'https://example.com'}
    >>> extract_hyperlinks('<html><body>No links here</body></html>')
    set()
    """
    if isinstance(text_html, pd.Series):
        return text_html.apply(extract_hyperlinks) # type: ignore
    
    if text_html is None:
        warnings.warn("`text_html` is None")
        return set()

    try:
        tree = html.fromstring(text_html)
    except:
        return set()
    
    links = tree.xpath('//a[@href]')
    
    urls = {link.get('href') for link in links}

    return urls

def __get_hyperlink_proportion(hyperlinks: list, word_count: int) -> float:
    """
    Private method for `get_hyperlink_proportion`.
    
    Parameters
    ----------
    hyperlinks : list
        List of hyperlinks found in the content.
    word_count : int
        Total number of words in the content.

    Returns
    -------
    float
        The proportion of hyperlinks to words, clipped between 0 and 1.
        Returns 0 if no hyperlinks are present, and 1 if word_count is 0.
    """
    if len(hyperlinks) <= 0:
        return 0
    
    if word_count == 0:
        return 1
    
    prop = len(hyperlinks) / word_count
    
    return np.clip(prop, 0, 1)

def get_hyperlink_proportion(hyperlinks: list | pd.Series, word_count: int | pd.Series) -> pd.Series | float:
    """
    Calculate the proportion of hyperlinks to word count.
    
    Parameters
    ----------
    hyperlinks : array-like or pandas.Series
        Hyperlinks extracted from text
    word_count : int or pandas.Series
        Number of words in the text
        
    Returns
    -------
    float or pandas.Series
        Proportion of hyperlinks to words, clipped to [0,1]
    """
    if isinstance(hyperlinks, pd.Series) and isinstance(word_count, pd.Series):
        if not hyperlinks.index.equals(word_count.index):
            raise ValueError("Series must have the same index")
        
        result = pd.Series(index=word_count.index, dtype=float)

        for i in hyperlinks.index:
            result[i] = __get_hyperlink_proportion(hyperlinks[i], word_count[i]) # type: ignore

        return result
    
    try:  
        return __get_hyperlink_proportion(hyperlinks, word_count) # type: ignore
    
    except:
        raise TypeError("Both `hyperlinks` and `word_count` should either be list/int or pandas.Series")