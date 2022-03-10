import click
import get_collocates


def sorting(countdict: dict, word_set:set) -> list[str]:
    pass


def compare(file1: str, file2: str) -> None:
    """Creats a csv with a comparison of the 0 and 2nd order adjs"""
    dict_0 = get_collocates.make_outdict(0, file1)
    dict_0_list = sorted(list(dict_0.keys()))
    dict_2 = get_collocates.make_outdict(2, file2)
    dict_2_list = sorted(list(dict_2.keys()))
    not_end = True
    with open('compare.csv', 'w', encoding='utf-8') as out_f:
        out_f.write('word_0, word_2, count_0, count_2, top_0, top_2, head_count_0, head_count_2, num_mutual, '
                    + 'mutual heads, num_unique_2, unique_2_heads\n')
        for w0, w2 in zip(dict_0_list, dict_2_list):
            # inp = input("Enter word pair (separated by space) or quit:")
            # if inp[0].lower() == 'q':
            #     not_end = False
            # # elif inp.split()[0] not in set(dict_0.keys) or inp.split()[1] not in set(dict_2.keys()):
            # #     print("invalid input")
            # else:
            w0_dict = dict_0[w0]
            w2_dict = dict_2[w2]
            out_f.write(w0 + ',' + w2 + ',')
            # out_f.write('------------------------------------------------\n')
            out_f.write(str(w0_dict['total_count']) + ',' + str(w2_dict['total_count']) + ',')
            w0_head_counts = list(w0_dict['head_counts'].items())
            w0_head_counts.sort(key=lambda y: y[1], reverse=True)
            w2_head_counts = list(w2_dict['head_counts'].items())
            w2_head_counts.sort(key=lambda y: y[1], reverse=True)
            out_f.write(str(w0_head_counts[0][0]) + ',' + str(w2_head_counts[0][0]) + ',')
            w0_head_set = set(w0_dict['head_counts'].keys())
            w2_head_set = set(w2_dict['head_counts'].keys())
            out_f.write(str(len(w0_head_set)) + ',' + str(len(w2_head_set)) + ',')
            intersection = w0_head_set & w2_head_set
            out_f.write(str(len(w0_head_set & w2_head_set)) + ',')
            mutual_heads = []
            for word in intersection:
                mutual_heads.append((word, w2_dict['head_counts'][word]))
            mutual_heads.sort(key=lambda y: y[1], reverse=True)
            # out_f.write('Mutual heads:\n')
            # mutual_set = w0_head_set & w2_head_set
            # mututal_heads = [word + for word in list(mutual_set)]
            mutual_heads = [x for x, y in mutual_heads]
            out_f.write(' '.join(mutual_heads) + ',')
            # out_f.write('Only 2nd Order heads')
            difference = w2_head_set - w0_head_set
            unique_heads = []
            for word in difference:
                unique_heads.append((word, w2_dict['head_counts'][word]))
            unique_heads.sort(key=lambda y: y[1], reverse=True)
            unique_heads = [x for x, y in unique_heads]
            out_f.write(str(len(difference)) + ',')
            out_f.write(' '.join(unique_heads))
            out_f.write('\n')


def summarize(file: str, order: int) -> None:
    """Creates a file with the summaries for each of the words in a particular order category"""
    word_dicts = get_collocates.make_outdict(order, file)
    filename = 'summarize_' + str(order) + '.csv'
    with open(filename, 'w', encoding='utf-8') as out_f:
        out_f.write('word, count, top, head_count, heads, texts\n')
        for w in word_dicts.keys():
            w_dict = word_dicts[w]
            out_f.write(w + ',')
            out_f.write(str(w_dict['total_count']) + ',')
            head_counts = list(w_dict['head_counts'].items())
            head_counts.sort(key=lambda y: y[1], reverse=True)
            if len(head_counts) > 0:
                out_f.write(str(head_counts[0][0]) + ',')
            else:
                out_f.write(',')
            head_set = set(w_dict['head_counts'].keys())
            out_f.write(str(len(head_set)) + ',')
            head_count_words = [x for (x, _) in head_counts[:100]]
            out_f.write(' '.join(head_count_words) + ',')
            out_f.write('\n')


@click.command()
@click.argument('command', type=str)
# @click.argument("file1", type=click.Path(exists=True))
# @click.argument("file2", type=click.Path(exists=True))
def main(command: str) -> None:
    if command == 'compare':
        file1 = input('File 1: ')
        file2 = input('File 2: ')
        compare(file1, file2)
    if command == 'summarize':
        file1 = input('File: ')
        order = input('Order: ')
        summarize(file1, int(order))


if __name__ == '__main__':
    main()
