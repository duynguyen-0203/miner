import argparse

import arguments
from src import utils
from src.trainer import Trainer


def _train(args):
    trainer = Trainer(args)
    trainer.train()


def _eval(args):
    trainer = Trainer(args)
    trainer.eval()


def main():
    parser = argparse.ArgumentParser(description='Arguments for Miner model', fromfile_prefix_chars='@',
                                     allow_abbrev=False)
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args
    subparsers = parser.add_subparsers(dest='mode', help='Mode of the process: train or test')

    train_parser = subparsers.add_parser('train', help='Training phase')
    arguments.add_train_arguments(train_parser)
    eval_parser = subparsers.add_parser('eval', help='Evaluation phase')
    arguments.add_eval_arguments(eval_parser)

    args = parser.parse_args()
    if args.mode == 'train':
        _train(args)
    elif args.mode == 'eval':
        _eval(args)


if __name__ == '__main__':
    main()
