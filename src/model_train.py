import argparse
from typing import Any

from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from model_utils import compute_metrics, MODEL_INFO, make_datasets

parser = argparse.ArgumentParser(description="Train the various models")
parser.add_argument('model', type=str, default='baseline', help='which model is this preprocessing for (default baseline)')
# parser.add_argument('--from_scratch', action='store_true', default=False, help='train all models from scratch')
parser.add_argument('lr', type=int, default=2e-5, help='learning rate')
parser.add_argument('epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('weight_decay', type=int, default=0.01, help='weight decay')
parser.add_argument('evaluation_strategy', type=str, default='epochs', help='evaluation strategy')
parser.add_argument('max_len', type=int, default=128, help='maximum length of input (should be lower for embed model)')


args = parser.parse_args()
DATAPATH = '../data/'
OUTPATH = '../model/'


def train_model_part(model_dict: dict):
    '''Training for the baseline and embeddings, where we want it to go all the way through the trainer'''
    gpu_avaliable = True if torch.cuda.is_available() else False
    training_args = TrainingArguments(
        OUTPATH + model_dict['model_dir'],
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        fp16=gpu_avaliable
    )
    if args.model == 'baseline':
        trainer = Trainer(
            model_dict['model'],
            training_args,
            train_dataset=model_dict['tokenized_datasets']['train'],
            eval_dataset=model_dict['tokenized_datasets']['dev'],
            compute_metrics=compute_metrics,
            tokenizer=model_dict['tokenizer']
        )
    else:  # embeddings aren't evaluated
        trainer = Trainer(
            model_dict['model'],
            training_args,
            train_dataset=model_dict['tokenized_datasets']['train'],
            tokenizer=model_dict['tokenizer']
        )

    print("Training...")
    trainer.train()


def main():
    model_name = args.model
    assert model_name in set(MODEL_INFO.keys()), "Invalid model name"  # make sure it is a valid model name
    model_dict = MODEL_INFO[model_name]
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    if type(tokenizer) == list:
        pass  # for the final model, when we want all the tokenizers
    filedir = model_dict['filepath']

    if model_name == 'baseline' or model_name == 'full_model':
        model_dict['dataset'] = load_dataset('csv', data_files={'train': DATAPATH + filedir + 'train.csv',
                                                                'dev': DATAPATH + filedir + 'dev.csv',
                                                                'test': DATAPATH + filedir + 'test.csv'})
    elif model_name == 'derived_embed':
        model_dict['dataset'] = load_dataset('csv', data_file=DATAPATH + 'train_embedding_data.csv')
    model_dict['tokenized_datasets'] = {}
    for name, data in model_dict['dataset'].items():
        model_dict['tokenized_datasets'][name] = make_datasets((name, data), tokenizer)
    if model_name == 'full_model':
        pass
    else:
        train_model_part(model_dict)


if __name__ == '__main__':
    main()
