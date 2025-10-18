import os
import re
import string
import pandas as pd

SPEAKER_TKN = "[A-Z][A-Z ]+"
SPEAKER_POSTFIX = ": "

SPEAKER_REG = re.compile(rf"({SPEAKER_TKN}{SPEAKER_POSTFIX})")
DASHDASH_REG = re.compile(r'--')
WHITESPACE_REG = re.compile(rf'[{string.whitespace}]')
REPEATING_WHITESPACE_REG = re.compile(rf'[{string.whitespace}][{string.whitespace}]+')
INSIDE_SQUARE_BRACKETS = re.compile(r'\[[^\]]+\]')

def join_speaker_lines(lines):
    return '\n'.join([f"{line['speaker']}: {line['line']}" for _, line in lines.iterrows()])

def get_speaker_lines(text):
    parts = SPEAKER_REG.split(text)[1:]  # first part is before first speaker, ignore
    if len(parts[0].strip()) == 0:
        parts = parts[1:]
    if len(parts)%2 == 1 and not SPEAKER_REG.search(parts[0]):
        parts = [None] + parts

    lines = []
    for i in range(0, len(parts) - 1, 2):
        lines.append([
            parts[i][:-2], #speaker
            parts[i + 1] #line
        ])
    return [[i] + l for i, l in enumerate(lines)]

def get_testimonies(testimonies_dir):
    testimonies = []
    for fname in os.listdir(testimonies_dir):
        if fname.endswith('.txt'):
            with open(testimonies_dir + os.sep + fname, 'r', encoding='utf-8') as f:
                testimony_id = fname.replace('.txt', '')
                testimony_content = f.read()
                testimonies.append([testimony_id, testimony_content])

    return pd.DataFrame(testimonies, columns=['testimony_id', 'content'])
