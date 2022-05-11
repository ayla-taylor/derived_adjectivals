import json
import spacy
import random
import pandas as pd


root_to_derived = {'cool': 'cooled',
                   'hard': 'hardened',
                   'black': 'blackened'}

spacy_nlp = spacy.load("en_core_web_sm",
                       exclude=["ner", "attribute_ruler", "lemmatizer"])


def read_sense_dict(filename: str) -> dict:
    with open(filename, 'r', encoding='utf8') as f:
        sense_dict = json.load(f)
    return sense_dict


def read_file(filename: str) -> dict:
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
    return data


def get_head(sent: str, target_word) -> str:
    parsed_doc = spacy_nlp(sent)
    bracketed_target = '[' + target_word + ']'
    target = ''
    for i, tok in enumerate(parsed_doc):
        if 0 < i < len(parsed_doc) - 1:
            bracketed_word = ''.join(parsed_doc[i-1:i+2].text)
            if bracketed_word == bracketed_target:
                target = tok.text
                head = tok.head.text
    return root_to_derived[target] + ' ' + head

