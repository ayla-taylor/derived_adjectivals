import json
import click
from collections import Counter, defaultdict


TARGET_FILES = ['root_adj_list.txt', 'root_verb_list.txt', 'double_derived_adj.txt']


def make_targetset(order: int) -> set[str]:
    filename = TARGET_FILES[order]
    targetset = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            targetset.add(line.strip())
    return targetset


def make_outdict(order: int, json_file: str):
    out_dict = dict()
    targetset = make_targetset(order)
    for target in targetset:
        out_dict[target] = {'total_count': 0, 'head_counts': defaultdict(int), 'texts': []}
    with open(json_file, 'r', encoding='utf8') as f:
        for line in f:
            word_dicts = json.loads(line)
            for i, w in enumerate(word_dicts):
                if w["text"] in targetset:
                    target_w = w['text']
                    out_dict[target_w]['total_count'] += 1
                    text = ''
                    for word in word_dicts[:i]:
                        text += word['text'] + ' '
                    text += '[' + w['text'] + ']' + ' '
                    for word in word_dicts[i + 1:]:
                        text += word['text'] + ' '
                    out_dict[target_w]['texts'].append(text)
                    head_index = w['head']
                    head = word_dicts[head_index - 1]
                    out_dict[target_w]['head_counts'][head['text']] += 1
                    # print(out_dict[target_w])
    dict_list = list(out_dict.items())
    dict_list.sort(key=lambda y: y[1]['total_count'], reverse=True)
    # for key, d in dict_list:
    #     print(key)
    #     print('count:', out_dict[key]['total_count'])
    #     head_counts = list(out_dict[key]['head_counts'].items())
    #     head_counts.sort(key=lambda y: y[1], reverse=True)
    #     print('head_counts:', str(head_counts).encode('utf-8'))
    # print(json.dumps(out_dict, sort_keys=False, indent=4))
    return out_dict


@click.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.argument("json_out", type=click.Path())
@click.argument("order", type=int)
@click.argument("target", type=str)
def main(json_file: str, json_out: str, order: int):
    out_dict = make_outdict(order, json_file)
    # print(head_dict)
    # print(nouns)
    # with open(json_out, 'w', encoding='utf8') as out_f:
    #     # print(noun_counts)
    #     for x, seqs in collocates.items():
    #         print(x, "|", seqs)
    #         out = x + " | " + seqs
    #         out_f.write(out)


if __name__ == '__main__':
    main()
