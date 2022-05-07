import random
import sys
import json
import spacy
import tqdm
import os

spacy_nlp = spacy.load("en_core_web_sm",
                       exclude=["ner", "attribute_ruler", "lemmatizer"])


def parse(data: dict, index: int):
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
                # print(sent)
                # print(tok.text, tok.dep_, tok.head.text, tok.head.pos_)
                if tok.tag_ == 'JJ':
                    # print(tok.text)
                    # print(i)
                    auto_tagged_indexes.append(i)
                else:
                    indexes.append(i)
        # print(sents)
        for j in indexes:
            # print([tok.text for tok in sent[:j]])
            sent_str = ' '.join([tok.text for tok in sent[:j]]) + ' [' + sent[j].text + '] ' + \
                       ' '.join([tok.text for tok in sent[j+1:]])
            # print(sent_str)
            sent_dict = {'index': index, 'data': sent_str}
            index += 1
            sents.append(sent_dict)
        for j in auto_tagged_indexes:
            # print([tok.text for tok in sent[:j]])
            sent_str = ' '.join([tok.text for tok in sent[:j]]) + ' [' + sent[j].text + '] ' + \
                       ' '.join([tok.text for tok in sent[j+1:]])
            # print(sent_str)
            sent_dict = {'index': index, 'data': sent_str}
            index += 1
            auto_sents.append(sent_dict)
    return sents, auto_sents, index
    # target_found = False
    # for sent in parsed_doc.sents:
    #     sent_str = ''
    #     target_sent = False
    #     extra_targets = 0
    #     for tok in sent:
    #         if tok.text == target_word:
    #             if not target_found:
    #                 sent_str += '[' + tok.text + '] '
    #                 target_sent = True
    #                 target_found = True
    #             else:
    #                 sent_str += tok.text + ' '
    #                 extra_targets += 1
    #         else:
    #             sent_str += tok.text + ' '
    #     sents.append((index, sent_str))
    #     index += 1
    #     previous = 1
    #     while extra_targets > 0:
    #         found = 0
    #         # print(extra_targets)
    #         target_found = False
    #         # sent_str += '\n '
    #         sent_str = ''
    #         for tok in sent:
    #             if tok.text == target_word:
    #                 found += 1
    #                 if previous >= found:
    #                     sent_str += tok.text + ' '
    #                 elif not target_found:
    #                     sent_str += '[' + tok.text + '] '
    #                     target_found = True
    #                     extra_targets -= 1
    #                     previous += 1
    #                 else:
    #                     sent_str += tok.text + ' '
    #             else:
    #                 sent_str += tok.text + ' '
    #     sents.append((index, sent_str))
    #     index += 1
    #     if target_sent:
    #         # return sent_str
    #         return sents, index


def create_files(filename: str, subfolder: str) -> None:
    print(f"processing {filename}")
    path = '../data/'
    outfile = filename[:-5] + "_annotation_ready.json"
    spacy_outfile = filename[:-5] + "_spacy.json"
    out_lines = []
    auto_lines = []
    # out_dict = {'data': {}}
    index = 0
    with open(path + subfolder + filename, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            data: dict = json.loads(line.strip())
            # print(type(data))
            str_list, auto_str, index = parse(data, index)
            for s in str_list:
                out_lines.append(s)
            for auto in auto_str:
                auto_lines.append(auto)
            # out_lines.append(parse(data))
    # outlines_uniq = list(set(out_lines))
    random.shuffle(out_lines)
    random.shuffle(auto_lines)
    with open(path + subfolder + outfile, "w", encoding='utf8') as out_f:
        # json.dump(out_dict, out_f)
        if len(out_lines) > 500:
            out_lines = out_lines[:500]
        json.dump(out_lines, out_f)

        #     out_f.write(line + '\n ')
    with open(path + subfolder + spacy_outfile, "w", encoding='utf8') as out_f:
        # json.dump(out_dict, out_f)
        if len(auto_lines) > 500:
            auto_lines = auto_lines[:500]
        json.dump(auto_lines, out_f)
        #     out_f.write(line + '\n ')


def main():
    path = '../data/'
    subfolder = sys.argv[-1]
    files = os.listdir(path + subfolder)
    # filename = 'double_derived/cooled.json'
    [create_files(file, subfolder) for file in files]


if __name__ == '__main__':
    main()
