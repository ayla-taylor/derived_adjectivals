import json
import click
from collections import Counter, defaultdict

TARGETS = {'blackened', 'cleaned', 'cleared', 'cooled', 'deepen', 'dirtied', 'dried', 'hardened', 'toughened',
           'enlarged', 'lengthened', 'reddened', 'sharpened', 'shortened', 'smoothed', 'softened', 'straightened',
           'strengthened', 'sweetened', 'tighten', 'warmed', 'weakened', 'whitened', 'widened'}


@click.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.argument("json_out", type=click.Path())
def main(json_file: str, json_out: str):
    collocates = defaultdict(list)
    nouns = []
    word_counts = defaultdict(int)
    head_dict = defaultdict(lambda: defaultdict(int))
    with open(json_file, 'r', encoding='utf8') as f:
        for line in f:
            word_dicts = json.loads(line)
            for i, w in enumerate(word_dicts):
                if w["text"] in TARGETS:
                    text = ''
                    for word in word_dicts[:i]:
                        text += word['text'] + ' '
                    text += '[' + w['text'] + ']' + ' '
                    for word in word_dicts[i+1:]:
                        text += word['text'] + ' '
                    head_index = w['head']
                    head = word_dicts[head_index - 1]
                    head_dict[w['text']][head['text']] += 1
                    nouns.append(head['text'])
                    seq = dict()
                    print(text)
                    print(head)
                    seq['text'] = text
                    # seq['full parse'] = word_dicts
                    seq['head'] = head['text']
                    # print(seq)
                    collocates[w['text']] += seq
                    # print(collocates)
    noun_counts = Counter(nouns)
    # print(head_dict)
    # print(nouns)
    with open(json_out, 'w', encoding='utf8') as out_f:
        # print(noun_counts)
        for x in collocates:
            # print(x)
            out_f.write(json.dumps(x))


if __name__ == '__main__':
    main()
