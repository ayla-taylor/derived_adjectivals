import json
import pandas as pd
import spacy
import random

DATAFILES = ['cooled-annotation-2_annotations.json', 'blackened-annotation_annotations.json']
SENSE_DICT = {'cooled': {'Sense 1': 'cool_1',
                       'Sense 2': 'cool_2'},
              'hardened': {'Sense 1': 'hard_1',
                       'Sense 2': 'hard_2',
                       'Sense 3': 'hard_3'},
              'blackened': {'Sense 1': 'black_1',
                        'Sense 2': 'black_2',
                        'Sense 3': 'black_3'},
              }

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
    sentence_dicts = []
    for datafile in DATAFILES:
        data = read_file(filepath + datafile)
        target_word = data['dataset']['name']
        sentences = data['examples']
        for sent in sentences:
            for annotation in sent['annotations']:
                if annotation['correct']:
                    head = annotation['value']
                    # data_sentence = sent['content']
                    # # truncated = parse(data_sentence, target_word, head)
                    for classification in sent['classifications']:
                        if classification['correct']:
                            tag = SENSE_DICT[target_word][classification['classname']]
                    data_dict = {'text': target_word + ' ' + head, 'label': tag}
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
    train = pd.DataFrame(sentence_dicts, columns=['text', 'label'])

    train.to_csv('../data/embedding_train.csv', index=False)


if __name__ == '__main__':
    main()
