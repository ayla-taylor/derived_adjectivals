import json
from collections import Counter, defaultdict

TARGET = 'dried'


def main(json_file: str, json_out: str):
    collocates = []
    nouns = []
    with open(json_file, encoding='utf8') as f:
        for line in f:
            word_dicts = json.loads(line)
            for i, w in enumerate(word_dicts):
                if w["text"] == TARGET:
                    if i > 0:
                        if word_dicts[i+1]['upos'] == 'NOUN':
                            next_w = word_dicts[i+1]['text']
                            nouns.append(next_w)
                        seq = dict()
                        seq['text'] = word_dicts[i-1]['text'] + ' ' + w['text'] + ' ' + word_dicts[i+1]['text']
                        seq['full parse'] = (word_dicts[i-1], w, word_dicts[i+1])
                        collocates.append(seq)
    noun_counts = Counter(nouns)
    print(nouns)
    with open(json_out, 'w', encoding='utf8') as out_f:
        print(noun_counts)
        for x in collocates:
            # print(x)
            out_f.write(json.dumps(x))


if __name__ == '__main__':
    main('extracted_mini.json', 'out.txt')
