# The point of this file is to get the data in the format for annotating
import random
import sys
import json
import spacy
import tqdm
import os

spacy_nlp = spacy.load("en_core_web_sm",
                       exclude=["ner", "attribute_ruler", "lemmatizer"])


def parse(data: dict, index: int) -> tuple:
    sents = []
    target_word = data['target_word']
    text = data['text']
    auto_sents = []
    parsed_doc = spacy_nlp(text)
    for sent in parsed_doc.sents:
        auto_tagged_indexes = []
        indexes = []
        for i, tok in enumerate(sent):
            if tok.text == target_word:
                print(sent)
                if tok.tag_ == 'JJ':
                    auto_tagged_indexes.append(i)
                else:
                    indexes.append(i)
        for j in indexes:
            sent_str = ' '.join([tok.text for tok in sent[:j]]) + ' [' + sent[j].text + '] ' + \
                       ' '.join([tok.text for tok in sent[j + 1:]])
            sent_dict = {'index': index, 'data': sent_str}
            index += 1
            sents.append(sent_dict)
        for j in auto_tagged_indexes:
            sent_str = ' '.join([tok.text for tok in sent[:j]]) + ' [' + sent[j].text + '] ' + \
                       ' '.join([tok.text for tok in sent[j + 1:]])
            sent_dict = {'index': index, 'data': sent_str}
            index += 1
            auto_sents.append(sent_dict)
    return sents, auto_sents, index


def create_files(filename: str, subfolder: str) -> None:
    """Create the files for the auto extracted data and the ones that are ready to be annotated"""
    print(f"processing {filename}")
    path = '../../data/pre_annotation/'
    outfile = filename[:-5] + "_annotation_ready.json"
    spacy_outfile = filename[:-5] + "_spacy.json"
    out_lines = []
    auto_lines = []
    index = 0
    # get data from files
    with open(path + subfolder + filename, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            data: dict = json.loads(line.strip())
            str_list, auto_str, index = parse(data, index)
            for s in str_list:
                out_lines.append(s)
            for auto in auto_str:
                auto_lines.append(auto)
    random.shuffle(out_lines)
    random.shuffle(auto_lines)

    # output into the two files
    with open(path + subfolder + outfile, "w", encoding='utf8') as out_f:
        if len(out_lines) > 500:
            out_lines = out_lines[:500]
        json.dump(out_lines, out_f)
    with open(path + subfolder + spacy_outfile, "w", encoding='utf8') as out_f:
        if len(auto_lines) > 500:
            auto_lines = auto_lines[:500]
        json.dump(auto_lines, out_f)


def main():
    path = '../../data/pre_annotation/'
    # the subfolder will either the category of target word
    subfolder = sys.argv[-1]
    files = os.listdir(path + subfolder)
    [create_files(file, subfolder) for file in files]


if __name__ == '__main__':
    main()
