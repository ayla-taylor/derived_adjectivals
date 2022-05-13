import json
import pandas as pd
import spacy
import random
from preprocessing_utils import derived_to_root, get_head

DATAFILES = ['cooled-sense-annotation-only_annotations.json',
             'blackened-sense-only_annotations.json',
             'hardened-sense-annotation_annotations.json']
# SENSE_DICT = {'cooled': {'Sense 1': '0',
#                          'Sense 2': '1'},
#               'hardened': {'Sense 1': '0',
#                            'Sense 2': '1',
#                            'Sense 3': '0'},
#               'blackened': {'Sense 1': '0',
#                             'Sense 2': '1',
#                             'Sense 3': '0'},
#               }
SENSE_DICT_FILE = "../data/sense_dict.json"


spacy_nlp = spacy.load("en_core_web_sm",
                       exclude=["ner", "attribute_ruler", "lemmatizer"])


def read_file(filename: str) -> dict:
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
    return data


# def parse(sentence: str, target_word: str, head_word: str) -> None:
#     truncated_sent = []
#     print(target_word, head_word)
#     parsed_doc = spacy_nlp(sentence)
#     print(parsed_doc)
#     for i, tok in enumerate(parsed_doc):
#         if tok.text.strip('-') == target_word:
#             target = tok
#         if tok.text.strip('-') == head_word:
#             head = tok
#     truncated_parse = parsed_doc[target.i:head.i+1] if target.i < head.i else []
#     print(truncated_parse)
#     for tok in truncated_parse:
#         if tok.text != ']' and tok.text != '[':
#             truncated_sent.append(tok.text)
#     print(truncated_sent)
#     return ' '.join(truncated_sent)

def main():
    filepath = '../data/Annotated/'
    with open(SENSE_DICT_FILE, 'r', encoding='utf8') as f:
        sense_dict = json.load(f)
    sentence_dicts = []
    for datafile in DATAFILES:
        data = read_file(filepath + datafile)
        if data['dataset']['name'].split()[0] in set(derived_to_root.keys()):
            target_word = data['dataset']['name'].split()[0]
        else:
            target_word = data['dataset']['name'].split()[-1]
        sentences = data['examples']
        for sent in sentences:
            data_sentence = sent['content']
            for annotation in sent['annotations']:
                if annotation['correct']:
                    data_dict = {}
                    head = annotation['value']
                    # data_sentence = sent['content']
                    # # truncated = parse(data_sentence, target_word, head)
            if sent['classifications']:
                for classification in sent['classifications']:
                    if classification['correct']:
                        # tag = SENSE_DICT[target_word][classification['classname']]
                        tag = classification['classname']
                        mini_data_dicts = []
                        for sense_tag, sense_def in sense_dict[derived_to_root[target_word]].items():
                            text2 = sense_def
                            if sense_tag == tag:
                                label = 1
                            else:
                                label = 0
                            mini_data_dicts.append({'text2': text2, 'label': label})
                        if sent['annotations']:
                            for annotation in sent['annotations']:
                                if annotation['correct']:
                                    head = annotation['value']
                                    text1 = target_word + ' ' + head
                        else:
                            text1 = ' '.join(get_head(data_sentence, target_word))
                        for data_dict in mini_data_dicts:
                            data_dict['text1'] = text1
                            sentence_dicts.append(data_dict)
                # words = []
                # for word in data_sentence.split():
                #     # if word == "[" + target_word + "]":
                #     #     words.append(word[1:-1])
                #     # else:
                #     #     words.append(word)
                #     words.append(word)
                # data_sentence = ' '.join(words)
                # tag = sent['classifications'][0]['classname']
                # for classification in sent['classifications']:
                #     if classification['correct']:
                #         tag = classification['classname']
                #         for sense_tag, sense_def in SENSE_DICT[target_word].items():
                #             if sense_tag == tag:
                #                 sentence_dicts.append({'text1': data_sentence, 'text2': sense_def, 'label': 1})
                #             else:
                #                 sentence_dicts.append({'text1': data_sentence, 'text2': sense_def, 'label': 0})
    random.shuffle(sentence_dicts)
    train = pd.DataFrame(sentence_dicts, columns=['text1', 'text2', 'label'])

    train.to_csv('../data/derived_embed/embedding_train.csv', index=False)


if __name__ == '__main__':
    main()
