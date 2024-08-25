import re
import numpy as np
from config.config import NUM_CLASSES, SAFE_IDX

def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    return regex.sub(_replacer, string)

def delete_newline_and_getters(string):
    string = string.replace('\n', '')
    regex_getter = r'function\s+(_get|get)[a-zA-Z0-9_]+\([^\)]*\)\s(external|view|override|virtual|private|returns|public|\s)*\(([^)]*)\)\s*\{(\r\n|\r|\n|\s)*return\s*[^\}]*\}'
    string = re.sub(regex_getter, '', string)
    return string

def one_hot_encode_label(label):
    one_hot = np.zeros(NUM_CLASSES)
    for elem in label:
        if elem < SAFE_IDX:
            one_hot[elem] = 1
        elif elem > SAFE_IDX:
            one_hot[elem - 1] = 1
    return one_hot

