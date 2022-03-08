import click
import get_collocates


@click.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
def main(file1: str, file2: str):
    dict_0 = get_collocates.make_outdict(0, file1)
    dict_2 = get_collocates.make_outdict(2, file2)
    not_end = True
    while not_end:
        inp = input("Enter word pair (separated by space) or quit:")
        if inp[0] == 'q':
            not_end = False
        # elif inp.split()[0] not in set(dict_0.keys) or inp.split()[1] not in set(dict_2.keys()):
        #     print("invalid input")
        else:
            w0, w2 = inp.split()
            w0_dict = dict_0[w0]
            w2_dict = dict_2[w2]
            print('words:\t\t', w0, '\t', w2)
            print('---------------------------------')
            print('Count:\t\t', w0_dict['total_count'], '\t', w2_dict['total_count'])
            w0_head_counts = list(w0_dict['head_counts'].items())
            w0_head_counts.sort(key=lambda y: y[1], reverse=True)
            w2_head_counts = list(w2_dict['head_counts'].items())
            w2_head_counts.sort(key=lambda y: y[1], reverse=True)
            print('Top head:\t', w0_head_counts[0][0], '\t', w2_head_counts[0][0])
            w0_head_set = set(w0_dict['head_counts'].keys())
            w2_head_set = set(w2_dict['head_counts'].keys())
            print('Mutual heads:')
            print(str(w0_head_set & w2_head_set).encode('utf-8'))
            print('Only 2nd Order heads:')
            print(w2_head_set - w0_head_set)
            print()


if __name__ == '__main__':
    main()
