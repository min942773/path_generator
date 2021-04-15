from random import seed, sample
import os
import json
import argparse

paths = {
    'train-statement': './data/{dataset}/statement/train.statement.jsonl',
    'dev-statement': './data/{dataset}/statement/dev.statement.jsonl',
    'train': './data/{dataset}/fairseq/inhouse/train.jsonl',
    'dev': './data/{dataset}/fairseq/inhouse/valid.jsonl',
    'test': './data/{dataset}/fairseq/inhouse/test.jsonl',
    'dict': './data/{dataset}/fairseq/inhouse/dict.txt',
}

default_num = {
    'obqa': 400,
    'csqa': 1241,
    'socialiqa': 1500,
    'small_csqa': 1241,
    'expanded_csqa': 1241,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='obqa', choices=['csqa', 'socialiqa', 'obqa', 'small_csqa', 'expanded_csqa'])
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--num_of_test', default=400, type=int)
    parser.add_argument('--qids_path', default='./data/obqa/inhouse_split_qids.txt')
    parser.add_argument('--qids_test_path', default='./data/obqa/inhouse_split_test_qids.txt') # EDITED
    args = parser.parse_args()

    parser.set_defaults(qids_path=f'./data/{args.dataset}/inhouse_split_qids.txt',
                        qids_test_path=f'./data/{args.dataset}/inhouse_split_test_qids.txt',
                        num_of_test=default_num[args.dataset])

    args = parser.parse_args()

    os.system(f'cp {paths["dev-statement"].format(dataset=args.dataset)} {paths["dev"].format(dataset=args.dataset)}')
    os.system(f'wget -nc -O {paths["dict"].format(dataset=args.dataset)} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt')

    total = sum(1 for _ in open(paths["train-statement"].format(dataset=args.dataset)))
    seed(args.seed)
    test_ids = set(sample(list(range(total)), args.num_of_test))
    print(args.qids_path)
    with open(paths["train-statement"].format(dataset=args.dataset), 'r', encoding='utf-8') as fin, \
            open(paths["train"].format(dataset=args.dataset), 'w', encoding='utf-8') as fout_train, \
            open(paths["test"].format(dataset=args.dataset), 'w', encoding='utf-8') as fout_test, \
            open(args.qids_path, 'w', encoding='utf-8') as fout_ids, \
            open(args.qids_test_path, 'w', encoding='utf-8') as fout_test_ids:
        for i, line in enumerate(fin):
            if i not in test_ids:
                dic = json.loads(line)
                fout_train.write(line)
                fout_ids.write(dic['id'] + '\n')
            else:
                dic = json.loads(line) # EDITED 이거때문에 test set이 사실상 잘못 만들어지고 있었음
                fout_test.write(line)
                fout_test_ids.write(dic['id'] + '\n')

    print(f'inhouse train set saved to {paths["train"].format(dataset=args.dataset)}')
    print(f'inhouse test set saved to {paths["test"].format(dataset=args.dataset)}')
    print(f'inhouse train ids saved to {args.qids_path}')
    print(f'inhouse test ids saved to {args.qids_test_path}')


if __name__ == '__main__':
    main()
