import re
def preprocess_text(text):
    # Use regular expression to match backslashes and everything before slash
    pattern = r'.*?/'
    # Replace matched pattern with an empty string
    result = re.sub(pattern, '', text)
    return result
