import pandas as pd
from typing import Union
import re
from ipaddress import ip_address
from array import array
from whois import whois
from functools import cache
import time
from collections import Counter
import warnings

def has_header_value(header_value: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    Check whether an email header (e.g., Received-SPF, DKIM-Signature, Authentication-Result) is present.

    Parameters
    ----------
    header_value : str or pandas.Series
        The header string or a Series of header strings.

    Returns
    -------
    bool or pandas.Series
        True or boolean Series indicating presence (non-null).

    Raises
    ------
    TypeError
        If the input is neither a string nor a pandas Series.

    Examples
    --------
    >>> has_header_value("v=spf1 include:_spf.google.com ~all")
    True

    >>> import pandas as pd
    >>> s = pd.Series(["v=spf1", None, ""])
    >>> has_header_value(s)
    0     True
    1    False
    2     True
    dtype: bool
    """
    if isinstance(header_value, pd.Series):
        return header_value.notnull()
    elif isinstance(header_value, str):
        return header_value is not None
    else:
        raise TypeError("Input must be a string or a pandas Series")
    
    
def has_dmarc_authentication(auth_result: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    Check whether the 'Authentication-Results' field contains a DMARC result.

    Parameters
    ----------
    auth_result : str or pandas.Series
        A string or Series representing the 'Authentication-Results' header.

    Returns
    -------
    bool or pandas.Series
        True or a boolean Series indicating whether 'dmarc=' is present 
        (case-insensitive). If input is a Series, returns a Series of booleans.
        If input is a string, returns a single boolean.

    Raises
    ------
    TypeError
        If the input is neither a string nor a pandas Series.

    Examples
    --------
    >>> has_dmarc_authentication("mx.google.com; dmarc=pass (p=NONE)")
    True

    >>> import pandas as pd
    >>> s = pd.Series(["spf=pass", "dmarc=fail", None])
    >>> has_dmarc_authentication(s)
    0    False
    1     True
    2    False
    dtype: bool
    """
    
    if isinstance(auth_result, pd.Series):
        return auth_result.str.contains('dmarc=', case=False, na=False)
    elif isinstance(auth_result, str):
        return 'dmarc=' in auth_result.lower()
    elif not auth_result:
        return False
    else:
        raise TypeError("Input must be a string or a pandas Series")


def get_dkim_result(auth_result: Union[str, pd.Series]) -> Union[str, pd.Series]:
    """
    Extract the DKIM result from the 'Authentication-Results' header.

    Parameters
    ----------
    auth_result : str or pandas.Series
        A string or Series representing the 'Authentication-Results' header.

    Returns
    -------
    str or pandas.Series
        The extracted DKIM result (e.g., 'pass', 'fail', 'neutral'). 
        Returns 'none' if no DKIM result is found.
        If input is a Series, returns a Series of strings.
        If input is a string, returns a single string.

    Raises
    ------
    TypeError
        If the input is neither a string nor a pandas Series.

    Examples
    --------
    >>> get_dkim_result("mx.google.com; dkim=pass header.d=example.com")
    'pass'

    >>> import pandas as pd
    >>> s = pd.Series([
    ...     "dkim=pass header.d=example.com", 
    ...     "spf=pass", 
    ...     None
    ... ])
    >>> get_dkim_result(s)
    0    pass
    1    none
    2    none
    dtype: object
    """
    
    if isinstance(auth_result, pd.Series):
        return auth_result.apply(lambda x: re.search(r'dkim=(\w+)', x).group(1).lower() if isinstance(x, str) and re.search(r'dkim=(\w+)', x) else 'none')
    elif isinstance(auth_result, str):
        dkim_result = re.search(r'dkim=(\w+)', auth_result)
        return dkim_result.group(1).lower() if dkim_result else 'none'
    elif not auth_result:
        return 'none'
    else:
        raise TypeError("Input must be a string or a pandas Series")
    
def get_spf_result(receive_spf: Union[str, pd.Series]) -> Union[str, pd.Series]:
    """
    Extract the SPF result from the 'Received-SPF' header.

    This function returns the first 5 characters of the input string, which typically
    correspond to the SPF result (e.g., "pass", "fail", "soft", "neutr"). If the input is not
    a string or is None, 'none' is returned.

    Parameters
    ----------
    receive_spf : str or pandas.Series
        A string or Series representing the 'Received-SPF' header from an email.

    Returns
    -------
    str or pandas.Series
        The extracted SPF result (typically one of: 'pass', 'fail', 'soft', 'neutr').
        Returns 'none' if the value is not a valid string.
        If input is a Series, returns a Series of results.
        If input is a string, returns a single result string.

    Raises
    ------
    TypeError
        If the input is neither a string nor a pandas Series.

    Examples
    --------
    >>> get_spf_result("pass (google.com: domain of example.com designates ...")
    'pass'

    >>> import pandas as pd
    >>> s = pd.Series([
    ...     "fail (google.com: domain of attacker.com ...)",
    ...     "softfail (google.com ...)",
    ...     None
    ... ])
    >>> get_spf_result(s)
    0     fail
    1    soft
    2    none
    dtype: object
    """
     
    if isinstance(receive_spf, pd.Series):
        return receive_spf.apply(lambda x: re.search(r'^(\w+)', x).group(0).lower() if isinstance(x, str) and re.search(r'^(\w+)', x) else 'none')
    elif isinstance(receive_spf, str):
        spf_result = re.search(r'^(\w+)', receive_spf).group(0).lower()
        return spf_result if spf_result else 'none'
    elif not receive_spf:
        return 'none'
    else:
        raise TypeError("Input must be a string or a pandas Series")
    
def get_dmarc_result(auth_result: Union[str, pd.Series]) -> Union[str, pd.Series]:
    """
    Extract the DMARC authentication result from the Authentication-Results header.

    Parameters
    ----------
    auth_result : str or pandas.Series
        The Authentication-Results header from the email. Can be a single string or a Series of strings.

    Returns
    -------
    str or pandas.Series
        The extracted DMARC result (`pass`, `fail`, etc.) as a string or a Series of strings.
        Returns 'none' if DMARC result is not found or if the input is not a valid string.

    Raises
    ------
    TypeError
        If the input is neither a string nor a pandas Series.

    Examples
    --------
    >>> get_dmarc_result("Authentication-Results: mx.google.com; dmarc=pass header.from=example.com")
    'pass'

    >>> import pandas as pd
    >>> s = pd.Series([
    ...     "dmarc=pass header.from=example.com",
    ...     "spf=pass smtp.mailfrom=example.com",
    ...     None
    ... ])
    >>> get_dmarc_result(s)
    0     pass
    1     none
    2     none
    dtype: object
    """
    
    if isinstance(auth_result, pd.Series):
        return auth_result.apply(lambda x: re.search(r'dmarc=(\w+)', x).group(1).lower() if isinstance(x, str) and re.search(r'dmarc=(\w+)', x) else 'none')
    elif isinstance(auth_result, str):
        dmarc_result = re.search(r'dmarc=(\w+)', auth_result)
        return dmarc_result.group(1).lower() if dmarc_result else 'none'
    elif not auth_result:
        return 'none'
    else:
        raise TypeError("Input must be a string or a pandas Series")

    
def extract_dkim_domain(dkim_signature: Union[str, pd.Series]) -> Union[str, pd.Series]:
    """
    Extract the domain from the DKIM-Signature header field.

    Parameters
    ----------
    dkim_signature : str or pandas.Series
        The DKIM-Signature header string or a Series of such strings from email headers.
        Expected to contain the `d=` tag that specifies the signing domain.

    Returns
    -------
    str or pandas.Series
        The domain extracted from the `d=` tag of the DKIM-Signature. Returns a lowercase string
        if input is a string, or a Series of lowercase strings (with None for unparsable/missing input) if input is a Series.

    Raises
    ------
    TypeError
        If the input is neither a string nor a pandas Series.

    Examples
    --------
    >>> extract_dkim_domain("v=1; a=rsa-sha256; d=example.com; s=selector1;")
    'example.com'

    >>> import pandas as pd
    >>> s = pd.Series([
    ...     "v=1; a=rsa-sha256; d=example.com; s=selector1;",
    ...     "v=1; a=rsa-sha256; d=another.org; s=selector2;",
    ...     None
    ... ])
    >>> extract_dkim_domain(s)
    0    example.com
    1    another.org
    2          None
    dtype: object
    """

    def extract(sig: str) -> Union[str, None]:
        if not isinstance(sig, str):
            return None
        match = re.search(r'd=([\w\.-]+)', sig)
        return match.group(1).lower() if match else None

    if isinstance(dkim_signature, pd.Series):
        return dkim_signature.apply(extract)
    elif isinstance(dkim_signature, str):
        return extract(dkim_signature)
    elif not dkim_signature:
        return None
    else:
        raise TypeError("dkim_signature must be either a string or a pandas Series.")
    

def dkim_domain_matches_sender(dkim_signature: Union[str, pd.Series], sender_domain: Union[str, pd.Series]) -> bool:
    """
    Compare the domain used in the DKIM-Signature header (`d=`) with the sender's domain.

    Parameters
    ----------
    dkim_signature : str or pandas.Series
        The DKIM-Signature header string or a Series of such strings from email headers.
        Expected to contain a `d=` tag that specifies the signing domain.

    sender_domain : str or pandas.Series
        The domain part of the sender's email address (e.g., the part after '@') or a Series of such domains.

    Returns
    -------
    bool or pandas.Series
        Returns True if the DKIM domain matches the sender domain (case-insensitive exact match), 
        or a Series of booleans for element-wise comparisons.

    Raises
    ------
    TypeError
        If the inputs are not both strings or both pandas Series of the same length.

    Examples
    --------
    >>> dkim_domain_matches_sender("v=1; a=rsa-sha256; d=example.com; s=selector1;", "example.com")
    True

    >>> import pandas as pd
    >>> sigs = pd.Series(["v=1; d=example.com;", "v=1; d=another.org;", None])
    >>> senders = pd.Series(["example.com", "another.org", "missing.com"])
    >>> dkim_domain_matches_sender(sigs, senders)
    0     True
    1     True
    2    False
    dtype: bool
    """
  
    if isinstance(dkim_signature, pd.Series) and isinstance(sender_domain, pd.Series):
        dkim_domains = extract_dkim_domain(dkim_signature)
        sender_domains = sender_domain.fillna('').str.lower()
        return dkim_domains == sender_domains

    elif isinstance(dkim_signature, str) and isinstance(sender_domain, str):
        dkim_domain = extract_dkim_domain(dkim_signature)
        return dkim_domain == sender_domain.lower() if dkim_domain else False
    
    elif not dkim_signature or not sender_domain:
        return False

    else:
        raise TypeError("Both inputs must be either str or pd.Series of the same length.")

def to_from_match(to_email: Union[str, pd.Series], from_email: Union[str, pd.Series]) -> Union[bool, pd.Series]: 
    """
    Checks whether the 'To' and 'From' email addresses match.

    Parameters
    ----------
    to_email : str or pd.Series
        The recipient email address or a Series of addresses.
    from_email : str or pd.Series
        The sender email address or a Series of addresses.

    Returns
    -------
    bool or pd.Series
        True if the email addresses match (case-insensitive), or Series of booleans if input is Series.

    Examples
    --------
    >>> to_from_match("alice@example.com", "Alice@Example.com")
    True

    >>> to_from_match(
    ...     pd.Series(["alice@example.com", "bob@example.com"]),
    ...     pd.Series(["ALICE@example.com", "eve@example.com"])
    ... )
    0     True
    1    False
    dtype: bool
    """
    
    if isinstance(to_email, pd.Series) and isinstance(from_email, pd.Series):
        return to_email.str.lower() == from_email.str.lower()
    
    elif isinstance(to_email, str) and isinstance(from_email, str):
        return to_email.lower() == from_email.lower()
    
    elif not to_email or not from_email:
        return False
    
    else:
        raise TypeError("Both inputs must be either str or pd.Series of equal length.")

def extract_spf_email(received_spf: Union[str, pd.Series]) -> Union[str, pd.Series]:
    """
    Extract the sender email from the `Received-SPF` header field.

    This function looks for the `envelope-from=` field inside the SPF result string and extracts the email address.

    Parameters
    ----------
    received_spf : str or pandas.Series
        The `Received-SPF` header string or a Series of such strings from email headers.
        Expected to contain the `envelope-from=` field.

    Returns
    -------
    str or pandas.Series
        The email extracted from the `envelope-from=` field. Returns a lowercase string if input is a string,
        or a Series of lowercase strings (with None for unparsable or missing entries) if input is a Series.

    Raises
    ------
    TypeError
        If the input is neither a string nor a pandas Series.

    Examples
    --------
    >>> extract_spf_email('pass (google.com: domain of test@example.com designates 1.2.3.4 as permitted sender) client-ip=1.2.3.4; envelope-from="test@example.com"; helo=mail.example.com;')
    'example.com'

    >>> import pandas as pd
    >>> s = pd.Series([
    ...     'pass (google.com: domain of test@example.com designates 1.2.3.4 as permitted sender) envelope-from="test@example.com";',
    ...     'neutral (spf=neutral) envelope-from="user@another.org";',
    ...     None
    ... ])
    >>> extract_spf_email
    0    example.com
    1    another.org
    2          None
    dtype: object
    """

    def extract(spf: str) -> Union[str, None]:
        if not isinstance(spf, str):
            return None
        match = re.search(r'envelope-from=["\']?([^"\'>\s]+)["\']?', spf)
        return match.group(1).lower() if match else None

    if isinstance(received_spf, pd.Series):
        return received_spf.apply(extract)
    elif isinstance(received_spf, str):
        return extract(received_spf)
    elif not received_spf:
        return None
    else:
        raise TypeError("receive_spf must be either a string or a pandas Series.")
    

def spf_email_matches_sender(received_spf: Union[str, pd.Series], sender_email: Union[str, pd.Series]) -> bool:
    """
    Compare the sender email address in the SPF record (`envelope-from=`) with the sender's email address.

    This function extracts the full email address from the SPF record's `envelope-from=` field
    and compares it to the provided sender email address. The comparison is case-insensitive 
    and supports both string inputs and pandas Series for batch evaluation.

    Parameters
    ----------
    received_spf : str or pandas.Series
        A single `Received-SPF` header string or a Series of such strings from email headers.

    sender_email : str or pandas.Series
        The sender's email address (e.g., "user@example.com") or a Series of such addresses.

    Returns
    -------
    bool or pandas.Series
        Returns True if the SPF `envelope-from=` email address matches the given sender email 
        (case-insensitive exact match), or a Series of booleans for element-wise comparison.

    Raises
    ------
    TypeError
        If the inputs are not both strings or both pandas Series of the same length.

    Examples
    --------
    >>> spf_email_matches_sender(
    ...     'pass (google.com: domain of test@example.com designates 1.2.3.4 as permitted sender) envelope-from="test@example.com";',
    ...     'test@example.com'
    ... )
    True

    >>> import pandas as pd
    >>> spf_headers = pd.Series([
    ...     'pass envelope-from="user@example.com";',
    ...     'pass envelope-from="admin@another.org";',
    ...     None
    ... ])
    >>> sender_emails = pd.Series(['user@example.com', 'admin@another.org', 'no-reply@other.com'])
    >>> spf_email_matches_sender(spf_headers, sender_emails)
    0     True
    1     True
    2    False
    dtype: bool
    """
    
    if isinstance(received_spf, pd.Series) and isinstance(sender_email, pd.Series):
        spf_domains = extract_spf_email(received_spf)
        sender_domains = sender_email.fillna('').str.lower()
        return spf_domains == sender_domains

    elif (not received_spf or isinstance(received_spf, str)) and isinstance(sender_email, str):
        spf_domain = extract_spf_email(received_spf)
        return spf_domain == sender_email.lower() if spf_domain else False

    else:
        raise TypeError("Both inputs must be either str or pd.Series of the same length.")
    
def check_different_reply_domain(from_domain: str | pd.Series | None, reply_to_domain: str | pd.Series | None) -> bool | pd.Series:
    """
    Check if the domain from 'From' header matches the domain from 'Reply-To' header.

    This function evaluates if there's a mismatch between the sender's domain in the 'From' 
    field and the domain in the 'Reply-To' field, which is a common indicator of phishing.

    Parameters
    ----------
    from_domain : str or pandas.Series or None
        Domain extracted from the 'From' email header field
    reply_to_domain : str or pandas.Series or None
        Domain extracted from the 'Reply-To' email header field

    Returns
    -------
    bool or pandas.Series of bool
        True if there's a suspicious configuration (different domains or missing fields),
        False if the configuration appears normal

    Notes
    -----
    Returns True (suspicious) in the following cases:
    - Reply-To exists but From doesn't
    - Both Reply-To and From are missing
    - Reply-To and From domains don't match
    """
    if isinstance(from_domain, pd.Series) and isinstance(reply_to_domain, pd.Series):
        result = pd.Series(False, index=from_domain.index)
        
        # Case where Reply-To domain doesn't exist
        no_reply_mask = reply_to_domain.isna() | (reply_to_domain == '')
        # Subcase: From exists (normal)
        has_from_mask = ~(from_domain.isna() | (from_domain == ''))
        
        # Both missing is suspicious
        result[no_reply_mask & ~has_from_mask] = True
        
        # Case where Reply-To domain exists
        has_reply_mask = ~no_reply_mask
        # Different domains is suspicious
        result[has_reply_mask & has_from_mask & (from_domain != reply_to_domain)] = True
        # Missing From but having Reply-To is suspicious
        result[has_reply_mask & ~has_from_mask] = True
        
        return result
    
    if not (isinstance(from_domain, str) and (not reply_to_domain or isinstance(reply_to_domain, str))):
        raise ValueError("`from_domain` and `reply_to_domain` must both be str or pandas.Series")
    
    # Original logic for string inputs
    # Case 1 & 2: Reply-To domain does not exist
    if not reply_to_domain:
        # If From exists, this is normal configuration
        if from_domain:
            return False
        # Both missing is suspicious
        else:
            return True
    
    # Case 3, 4 & 5: Reply-To domain exists
    else:
        # From domain exists
        if from_domain:
            # Match is normal
            if reply_to_domain == from_domain:
                return False
            # Different domains is suspicious
            else:
                return True
        # Missing From but having Reply-To is suspicious
        else:
            return True
        
def extract_first_server_transfer(received_headers: list | pd.Series | array | None) -> str | pd.Series | None:
    """
    Extract the first meaningful 'Received' header that captures the server-to-server transfer.

    This function identifies the first server-to-server email transfer from external
    to UBC mail systems, which is useful for analyzing the email's origin.

    Parameters
    ----------
    received_headers : list or pandas.Series or array.array or None
        A list of 'Received' headers from an email or collection of emails,
        usually ordered from most recent (index 0) to oldest (index n)

    Returns
    -------
    str or pandas.Series or None
        The identified first server-to-server transfer header, or None if not found
        - If input is a list: Returns a string with the first relevant header
        - If input is a Series: Returns a Series with processed headers
        
    Notes
    -----
    The function prioritizes finding:
    1. Headers showing the UBC mail relay connection (indicating an incoming email)
    2. Headers with external IP addresses (often indicating external connections)
    3. Any header with both "from" and "by" components that isn't internal
    """
    if isinstance(received_headers, pd.Series):
        return received_headers.apply(extract_first_server_transfer) # type: ignore
    
    if not isinstance(received_headers, list) and hasattr(received_headers, 'tolist'):
        received_headers = received_headers.tolist() # type: ignore

    if not received_headers:
        warnings.warn("None entry found in `received_headers`")
        return None
    
    if not isinstance(received_headers, list):
        raise ValueError("`received_headers` must either be a list or able to be converted to a list")
    
    # Reverse the headers to start from the earliest transfer
    for header in reversed(received_headers):
        if not isinstance(header, str):
            continue
            
        # Skip internal transfers and known patterns for local processing
        if "localhost" in header or "127.0.0.1" in header or \
           "with mapi id" in header or \
           "envelope-from" in header:
            continue

        # Look for the UBC mail relay pattern
        if re.search(r'by\s+[a-zA-Z0-9-]+\.mail-relay\.ubc\.ca', header):
            return header
        
        # Look for headers with external IP addresses
        # This pattern matches "unknown [IP]" which often indicates external connections
        if re.search(r'unknown \[\d+\.\d+\.\d+\.\d+\]', header):
            return header
        
        # Also match any "from" that has a real IP that's not localhost
        if "from" in header and "by" in header:
            ip_matches = re.findall(r'\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]', header)
            for ip in ip_matches:
                address = ip_address(ip)
                if address.is_global:
                    return header
    
    # If nothing matched our criteria, use the first header with from and by
    for header in reversed(received_headers):
        # Skip internal transfers and known patterns for local processing
        if "localhost" in header or "127.0.0.1" in header or \
           "with mapi id" in header or \
           "envelope-from" in header:
            continue

        if isinstance(header, str) and "from" in header and "by" in header:
            return header
    
    return None

def extract_domain_from_received_header(received_header: str | pd.Series | None) -> str | pd.Series | None:
    """
    Extract domain name from 'From' field in Received email header.
    
    This function parses a Received email header to extract the domain name
    of the sending server, which helps establish the source of the email.
    
    Parameters
    ----------
    received_header : str or pandas.Series or None
        The Received header field containing information about email routing
        
    Returns
    -------
    str, pandas.Series or None
        The extracted domain name of the sending mail server
        - If input is a string: Returns domain as string or None if not found
        - If input is a Series: Returns a Series of domain names
        
    Examples
    --------
    >>> header = "from mail-server.example.com ([192.168.1.1]) by receiver.com"
    >>> extract_domain_from_received_header(header)
    'mail-server.example.com'
    """
    if isinstance(received_header, pd.Series):
        return received_header.apply(extract_domain_from_received_header) # type: ignore

    if not received_header:
        warnings.warn("None entry found in `received_header`")
        return None
    
    if not isinstance(received_header, str):
        raise ValueError("`received_header` must be str or pandas.Series or None")
        
    domain_pattern = r'from\s+([a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z0-9][-a-zA-Z0-9.]*)[\s\(]'
    
    match = re.search(domain_pattern, received_header)
    if match:
        return match.group(1)
    
    fallback_pattern = r'from\s+[^\(]*\([^H]*HELO\s+([a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z0-9][-a-zA-Z0-9.]*)'
    
    match = re.search(fallback_pattern, received_header)
    if match:
        return match.group(1)
    
    return None

def extract_ip_from_received_header(received_header: str | pd.Series | None) -> str | pd.Series | None:
    """
    Extract IP address from 'Received' header in an email.

    This function identifies and extracts the IP address of the sending mail server
    from the 'Received' header, useful for analyzing the true origin of an email.

    Parameters
    ----------
    received_header : str or pandas.Series or None
        The Received header string containing information about email routing,
        or a Series of such headers
        
    Returns
    -------
    str, pandas.Series or None
        The extracted IP address of the sending server
        - If input is a string: Returns the first IP address found or None
        - If input is a Series: Returns a Series of extracted IP addresses
        
    Examples
    --------
    >>> header = "from mail-server.example.com ([192.168.1.1]) by receiver.com"
    >>> extract_ip_from_received_header(header)
    '192.168.1.1'
    """
    if isinstance(received_header, pd.Series):
        return received_header.apply(extract_ip_from_received_header) # type: ignore

    if not received_header:
        warnings.warn("None entry found in `received_header`")
        return None
    
    if not isinstance(received_header, str):
        raise ValueError("`received_header` must be str or pandas.Series or None")
    
    if isinstance(received_header, str):
        # Extract IPv4 addresses (looking for standard dotted decimal format in brackets)
        ipv4_pattern = r'\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]'
        ipv4_matches = re.findall(ipv4_pattern, received_header)
        
        if ipv4_matches:
            return ipv4_matches[0]
        
        # Extract IPv6 addresses (various formats)
        # Standard IPv6 format with colons in brackets
        ipv6_pattern1 = r'\[([0-9a-fA-F:]+:[0-9a-fA-F:]+)\]'
        # IPv6 with port specification
        ipv6_pattern2 = r'([0-9a-fA-F:]+:[0-9a-fA-F:]+)\)'
        # IPv6 with HELO
        ipv6_pattern3 = r'\(([0-9a-fA-F:]+:[0-9a-fA-F:]+)\)'
        
        ipv6_matches = []
        ipv6_matches.extend(re.findall(ipv6_pattern1, received_header))
        ipv6_matches.extend(re.findall(ipv6_pattern2, received_header))
        ipv6_matches.extend(re.findall(ipv6_pattern3, received_header))
        
        # Filter out non-IP-looking matches and add valid IPv6 addresses
        for match in ipv6_matches:
            if ':' in match and match.count(':') >= 2:  # Simple validation for IPv6
                return match
        
        return None
    
    if not received_header or not isinstance(received_header, list):
        return 
    
    result = []
    
    for header in received_header:
        if not isinstance(header, str):
            result.append(None)
            continue
            
        # Extract IPv4 addresses
        ipv4_pattern = r'\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]'
        ipv4_matches = re.findall(ipv4_pattern, header)
        
        if ipv4_matches:
            result.append(ipv4_matches[0])
            continue
        
        # Extract IPv6 addresses (various formats)
        ipv6_pattern1 = r'\[([0-9a-fA-F:]+:[0-9a-fA-F:]+)\]'
        ipv6_pattern2 = r'([0-9a-fA-F:]+:[0-9a-fA-F:]+)\)'
        ipv6_pattern3 = r'\(([0-9a-fA-F:]+:[0-9a-fA-F:]+)\)'
        
        ipv6_matches = []
        ipv6_matches.extend(re.findall(ipv6_pattern1, header))
        ipv6_matches.extend(re.findall(ipv6_pattern2, header))
        ipv6_matches.extend(re.findall(ipv6_pattern3, header))
        
        for match in ipv6_matches:
            if ':' in match and match.count(':') >= 2:
                result.append(match)
                break
        else:  # No IPv6 match found
            result.append(None)
    
    return result if result else None

def check_email_from_ubc(received_header: str | pd.Series | None) -> bool | pd.Series:
    """
    Check if an email originates from UBC based on received header.

    This function identifies emails that originate from UBC's email infrastructure,
    which can help distinguish between legitimate internal emails and external phishing
    attempts masquerading as internal communications.

    Parameters
    ----------
    received_header : str or pandas.Series or None
        The first meaningful 'Received' header from an email or collection of emails
        
    Returns
    -------
    bool or pandas.Series
        True if the email originates from a UBC mail server, False otherwise
        - If input is a string: Returns a boolean indicating UBC origin
        - If input is a Series: Returns a Series of boolean values
        
    Examples
    --------
    >>> received = "from mail-server-1.ubc.ca (10.10.10.10) by mail-server-0.ubc.ca"
    >>> check_email_from_ubc(received)
    True
    """
    if isinstance(received_header, pd.Series):
        return received_header.apply(check_email_from_ubc) # type: ignore
        
    if not received_header:
        warnings.warn("None entry found in `received_header`")
        return False
    
    if not isinstance(received_header, str):
        raise ValueError("`received_header` must be str or pandas.Series or None")
    
    pattern = r'from\s+[^\s]+\.ubc\.ca\b'
    match = re.search(pattern, received_header, re.IGNORECASE)
    
    return match is not None

@cache
def __get_name_servers(domain: str) -> list | None:
    """
    Private method for `get_name_servers`.

    Parameters
    ----------
    domain : str
        Domain name to query for name servers
        
    Returns
    -------
    list or None
        List of name servers for the domain,
        or None if lookup fails or domain doesn't exist
    """    
    if not domain:
        warnings.warn("None entry found in `domain`")
        return None
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            result = whois(domain)
            if 'name_servers' in result:
                return result['name_servers']
        except Exception as e:
            if attempt == max_attempts - 1:
                return None
        time.sleep(0.5)
    
    return None

def get_name_servers(domain: str | pd.Series | None) -> list | pd.Series | None:
    """
    Extract name servers for a domain using Python's whois library.

    This function looks up DNS name servers for a domain name, which helps verify 
    the authenticity of domain ownership and identify mismatches that might 
    indicate spoofing attempts.

    Parameters
    ----------
    domain : str or pandas.Series
        Domain name(s) to query for name servers
        
    Returns
    -------
    list, pandas.Series, or None
        List of name servers for the domain, Series of lists for multiple domains,
        or None if lookup fails or domain doesn't exist

    Examples
    --------
    >>> get_name_servers('google.com')
    ['ns1.google.com', 'ns2.google.com', 'ns3.google.com', 'ns4.google.com']
    >>> get_name_servers('nonexistentdomain123456789.com')
    None
    """
    if isinstance(domain, pd.Series):
        return domain.apply(lambda x: __get_name_servers(x))   # type: ignore
    
    return __get_name_servers(domain)

def check_sender_name_servers_match(sender_from_name_servers: list | pd.Series | None, 
                                    sender_received_name_servers: list | pd.Series | None) -> bool | pd.Series:
    """
    Check if name servers from sender domain (based on 'From') match name servers from 'Received' domain.
    
    Parameters
    ----------
    sender_from_name_servers : list, pandas.Series, or None
        List of name servers from the sender's domain
    sender_received_name_servers : list, pandas.Series, or None
        List of name servers from the received message domain
        
    Returns
    -------
    bool or pandas.Series
        True if name servers match, False otherwise
    """
    if isinstance(sender_from_name_servers, pd.Series) and isinstance(sender_received_name_servers, pd.Series):
        matches = pd.Series(False, index=sender_from_name_servers.index)
        
        for idx in sender_from_name_servers.index:
            try:
                from_ns = sender_from_name_servers.loc[idx]
                received_ns = sender_received_name_servers.loc[idx]
                
                if not from_ns or not received_ns:
                    matches[idx] = False
                    continue
                    
                matches[idx] = Counter(from_ns) == Counter(received_ns)
            except Exception as e:
                matches[idx] = False
                
        return matches
    
    if hasattr(sender_from_name_servers, 'tolist'):
        sender_from_name_servers = sender_from_name_servers.tolist()
    if hasattr(sender_received_name_servers, 'tolist'):
        sender_received_name_servers = sender_received_name_servers.tolist()
    
    if not sender_from_name_servers or not sender_received_name_servers:
        return False
    
    if not (isinstance(sender_from_name_servers, list) and isinstance(sender_received_name_servers, list)):
        raise ValueError("Both inputs must be a list or coercible to a list")
    
    return Counter(sender_from_name_servers) == Counter(sender_received_name_servers)

def get_routing_before_ubc(received_headers: list | pd.Series) -> list | pd.Series:
    """
    Extract routing headers before UBC mail servers.

    This function filters out the received headers to identify server transfer paths
    before reaching UBC's mail infrastructure, which helps analyze the origin
    and path of potentially suspicious emails.

    Parameters
    ----------
    received_headers : list or pandas.Series
        List of received headers from an email or Series of such lists

    Returns
    -------
    list or pandas.Series
        Headers that indicate routing path before entering UBC mail servers
        - If input is a list: Returns filtered list of received headers
        - If input is a Series: Returns Series with filtered headers for each email
    """
    if isinstance(received_headers, pd.Series):
        return received_headers.apply(get_routing_before_ubc)   # type: ignore

    domain_pattern = r'from\s+[^\s]+\.ubc\.ca\b'

    servers = []
    
    external = False
    if received_headers is not None:
        for header in received_headers:
            match = re.search(domain_pattern, header)
            if not match:
                external = True

            if external == True:
                servers.append(header)

    return servers

def get_routing_length_before_ubc(received_headers: list | pd.Series) -> list | pd.Series:
    """
    Get routing length before UBC mail servers.

    This function filters out the received headers to identify the numbers of server transfers
    before reaching UBC's mail infrastructure, which helps analyze the origin
    and path of potentially suspicious emails.

    Parameters
    ----------
    received_headers : list or pandas.Series
        List of received headers from an email or Series of such lists

    Returns
    -------
    int or pandas.Series
        Number of server transfers before reaching UBC's mail servers
        - If input is a list: Returns number of server transfers
        - If input is a Series: Returns Series with number of server transfers for each email
    """
    if isinstance(received_headers, pd.Series):
        return received_headers.apply(get_routing_length_before_ubc)   # type: ignore

    if isinstance(received_headers, list):
        return len(get_routing_before_ubc(received_headers))
    
    return 0

def get_internal_server_transfer_count(received_header: list | pd.Series) -> int | pd.Series:
    """
    Count the number of internal server transfers in the email routing path.

    This function analyzes the received headers to identify how many internal server 
    transfers occurred before reaching the email's final destination, which helps 
    detect potential internal mail server abuse or unusual routing patterns.

    Parameters
    ----------
    received_header : list or pandas.Series
        List of received headers from an email or Series of such lists

    Returns
    -------
    int or pandas.Series
        The number of internal server transfers detected
        - If input is a list: Returns integer count of internal transfers
        - If input is a Series: Returns Series of counts for each email
        
    Examples
    --------
    >>> headers = ["from internal.example.com ([192.168.1.1])", "from mail.external.com ([203.0.113.1])"]
    >>> get_internal_server_transfer_count(headers)
    1
    """
    if isinstance(received_header, pd.Series):
        return received_header.apply(get_internal_server_transfer_count)    # type: ignore
    
    if not isinstance(received_header, list):
        return 0

    results = []
    for header in received_header:
        try:
            ip = ip_address(extract_ip_from_received_header(header))    # type: ignore
            results.append(ip.is_private)
        except:
            results.append(False)

    return sum(results)


def check_url_present_subject(subject: str | pd.Series) -> bool | pd.Series:
    """
    Check if a URL is present in the given subject.

    Parameters
    ----------
    subject : str or pandas.Series
        The text or texts to check for URLs.

    Returns
    -------
    bool or pandas.Series
        True if a URL is found in the subject, False otherwise.
        If input is a pandas Series, returns a Series of boolean values.
        
    Notes
    -----
    URLs are detected using a regular expression pattern that looks for:
    - URLs with explicit protocols (http://, https://, ftp://)
    - URLs starting with 'www.'
    - Domain-like patterns (e.g., example.com)
    """

    if isinstance(subject, pd.Series):
        return subject.apply(check_url_present_subject)

    if not subject:
        return False

    url_pattern = re.compile(
        r'(?:(?:https?://|ftp://|www\.)[^\s]+|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?)'
    )
    urls = url_pattern.findall(subject)

    if urls and len(urls) > 0:
        return True
    
    return False