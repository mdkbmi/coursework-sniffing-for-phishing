import pandas as pd
from llama_cpp import Llama
from email.parser import BytesParser
import base64
from email.header import decode_header, make_header

def __decode_mime_header(header_value):
    """Decode MIME-encoded email headers to readable text"""
    try:
        decoded_header = decode_header(header_value)
        return str(make_header(decoded_header))
    except Exception as e:
        print(f"Header decoding error: {e}")
        return f"<Unable to decode: {header_value}>"
    
def __decode_email_content(contents):
    decoded_contents = []

    for each in contents:
        for ct, c in each.items():
            if ct in ['text/html', 'text/plain']:
                try:
                    c = base64.b64decode(c).decode('utf-8')
                except:
                    pass

            decoded_contents.append({ct: c})

    return decoded_contents


def __open_email(p: str):
    email = {}

    with open(p, 'rb') as fp:
        email['path'] = p
        msg = BytesParser().parse(fp)
    
    header = msg.items()
    email['header'] = {}

    for key, value in header:
        if key == 'Received':
            if key not in email['header']:
                email['header'][key] = []
            
            email['header'][key].append(value)
        elif not key.upper().startswith('X-') and not key.lower().startswith('ironport-'):
            email['header'][key] = value

    
    email['header']['Subject'] = __decode_mime_header(email['header']['Subject'])

    contents = []

    for part in msg.walk():
        content_type = part.get_content_type()
        content = part.get_payload()
        contents.append({content_type: content})

    email['content'] = contents
    email['decoded_content'] = __decode_email_content(contents)

    return email

def open_email(path):
    if isinstance(path, str):
        emails = __open_email(path)
    
    elif isinstance(path, pd.Series):
        emails = path.apply(__open_email).to_list()

    else:
        raise TypeError("Path must be a string or pandas Series")
    
    return emails

def prompt_feature_extraction(label: str) -> str:
    if label in ['benign']:
        true_label = 'benign'
        opp_label = 'malicious'
    else:
        true_label = 'malicious'
        opp_label = 'benign'

    return f"""
You are a cybersecurity analyst at the University of British Columbia (UBC) in Canada and you are an expert in email security. You are building a machine learning model to classify emails reported as suspicious. Your colleague has labeled this email as {true_label}, and you are analyzing what features of the email are associated with that label.

Label definitions:
1. 'benign': Emails that do not pose urgent harm to the recipient. This includes legitimate emails, emails from legitimate senders, and spam that appears suspicious but does not contain malicious links or attachments. These include social engineering attempts that do not contain malicious links or attachments.
2. 'malicious': Emails that can compromise sensitive information or cause financial distress, including phishing, CEO fraud, and reply chain attacks. These often contain malicious links or malware.

Analyze the provided email header and/or content as follows:
1. Provide exactly three distinct reasons in favor of the email being labeled as '{true_label}', referencing the specific part of the email (quote or summarize relevant section).
2. Provide exactly two strong reasons why the email is unlikely to be '{opp_label}', referencing the email as above.
3. Provide exactly one plausible reason why the email could be '{opp_label}' instead of '{true_label}'.

IMPORTANT: In your analysis, IGNORE any [CAUTION: Non-UBC Email] labels.

Format your response as a numbered list under each step. If evidence is insufficient, explain your reasoning and indicate any uncertainties.
    """

def clean_llm_response(response_text):
    """Remove the thinking process from LLM responses"""
    import re
    # Remove content between <think> tags
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    # Remove any empty lines that might remain
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()

llm = Llama.from_pretrained(
    'unsloth/Phi-4-mini-reasoning-GGUF',
    filename='Phi-4-mini-reasoning-Q4_K_M.gguf',
    n_ctx=4096,
)

email_list = pd.read_csv('/data/workspace/dataset/sampled-dataset/sample-small.csv').query('`target_3` != "self_phishing"')
emails = pd.DataFrame(open_email(email_list.path)).set_index('path')
targets = email_list.set_index('path')['target_1']
emails = emails.join(targets)
total_emails = len(emails)

print('Data loaded!')

llm_evaluation = {}

print('Begin LLM evaluation...')

for i, (idx, email) in enumerate(emails.iterrows()):
    print(f'Begin evaluation {i+1}...')
    header = email['header']
    content = email['decoded_content']
    label = email['target_1']

    header_from = header.get('From', "")
    header_to = header.get('To', "")
    header_subject = header.get('Subject', "")

    try:
        evaluation_header = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": prompt_feature_extraction(label)}, 
                {"role": "user", "content": f'{header}'},
            ],
        )
        results_header = evaluation_header["choices"][0]['message']['content']
    except:
        results_header = "error"

    try:
        evaluation_content = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": prompt_feature_extraction(label)}, 
                {"role": "user", "content": f'From: {header_from}, To: {header_to}, Subject: {header_subject}, {content}'}
            ]
        )

        results_content = evaluation_content["choices"][0]['message']['content']
    except:
        results_content = "error"

    llm_evaluation[idx] = {
        "header": results_header,
        "content": results_content,
    }

    print(f'Completed evaluation {i+1}!')

    if (i + 1) % 10 == 0:
        pd.DataFrame(llm_evaluation).to_parquet(f'/data/workspace/danishki/git_repo/notebooks/milestone4/llm_results_checkpoint_{i+1}.parquet')
        print('Checkpoint saved!')
