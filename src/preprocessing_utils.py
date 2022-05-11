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
    print(parsed_doc)
    bracketed_target = '[' + target_word + ']'
    for i, tok in enumerate(parsed_doc[1:-1]):
        bracketed_word = ''.join(parsed_doc[i-1:i+1])
        if bracketed_word == bracketed_target:
            target = tok
            head = target.head
    return root_to_derived[target.text] + ' ' + head.text

