import argparse
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForNextSentencePrediction, \
    AutoModelForSequenceClassification, BertModel, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

from model_utils import compute_metrics, MODEL_INFO, make_datasets, DenseModel

parser = argparse.ArgumentParser(description="Train the various models")
parser.add_argument('--model', type=str, default='baseline',
                    help='which model is this preprocessing for (default baseline)')
parser.add_argument('--baseline_checkpoint', type=str, help='the number of the checkpoint for the baseline model')
parser.add_argument('--embed_checkpoint', type=str, help='the number of the checkpoint for the embed model')

# parser.add_argument('--from_scratch', action='store_true', default=False, help='train all models from scratch')
parser.add_argument('--lr', type=int, default=2e-5, help='learning rate')
parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--weight_decay', type=int, default=0.01, help='weight decay')
parser.add_argument('--evaluation_strategy', type=str, default='epoch', help='evaluation strategy')
parser.add_argument('--max_len', type=int, default=128,
                    help='maximum length of input (should be lower for embed model)')

args = parser.parse_args()
DATAPATH = '../data/'
OUTPATH = '../model/'


def train_model_part(model_dict: dict):
    """Training for the baseline and embeddings, where we want it to go all the way through the trainer"""
    gpu_avaliable = True if torch.cuda.is_available() else False
    training_args = TrainingArguments(
        OUTPATH + model_dict['model_dir'],
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.evaluation_strategy,
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
        print(model_dict['tokenized_datasets']['train'])
        trainer = Trainer(
            model_dict['model'],
            training_args,
            train_dataset=model_dict['tokenized_datasets']['train'],
            eval_dataset=model_dict['tokenized_datasets']['train'],
            compute_metrics=compute_metrics,
            tokenizer=model_dict['tokenizer']
        )

    print("Training...")
    trainer.train()


def train_full_model(model_dict: dict) -> None:
    # for split in ['train', 'dev', 'test']:
    # gpu_avaliable = True if torch.cuda.is_available() else False
    datafile = '../data/full_model/train.csv'
    eval_file = '../data/full_model/dev.csv'
    df = pd.read_csv(datafile)
    text1 = df['text1'].tolist()
    text2 = df['text2'].tolist()
    derived_pairs = df['text2'].tolist()
    label_1d = torch.tensor(df['label'])
    # torch.save(label_1d, 'labels.pt')
    label_list = []
    for label in df['label']:
        if label == 0:
            label_list.append([1, 0])
        else:
            label_list.append([0,1])
    labels = torch.tensor(label_list)
    torch.save(labels, 'labels.pt')

    eval_file = '../data/full_model/dev.csv'
    eval_df = pd.read_csv(eval_file)
    eval_text1 = eval_df['text1'].tolist()
    eval_text2 = eval_df['text2'].tolist()
    eval_derived_pairs = eval_df['text2'].tolist()
    eval_label_1d = torch.tensor(eval_df['label'])
    # torch.save(label_1d, 'labels.pt')
    eval_label_list = []
    for label in eval_df['label']:
        if label == 0:
            eval_label_list.append([1, 0])
        else:
            eval_label_list.append([0,1])
    eval_labels = torch.tensor(label_list)
    torch.save(eval_labels, 'eval_labels.pt')


    print('initializing models and tokenizers....')
    baseline_model_name, embed_model_name = model_dict['base_model']
    baseline_model_name = baseline_model_name + args.baseline_checkpoint
    embed_model_name = embed_model_name + args.embed_checkpoint

    baseline_tokenizer = BertTokenizer.from_pretrained(baseline_model_name)
    baseline_model = BertModel.from_pretrained(baseline_model_name)

    embed_tokenizer = BertTokenizer.from_pretrained(embed_model_name)
    embed_model = BertModel.from_pretrained(embed_model_name)

    print("tokenixing...")
    inputs_base = baseline_tokenizer(text1, text2, padding=True, truncation=True, return_tensors='pt')
    inputs_embeds = embed_tokenizer(derived_pairs, padding=True, truncation=True, return_tensors='pt')

    print("tokenixing...")
    eval_inputs_base = baseline_tokenizer(eval_text1, eval_text2, padding=True, truncation=True, return_tensors='pt')
    eval_inputs_embeds = embed_tokenizer(eval_derived_pairs, padding=True, truncation=True, return_tensors='pt')

    # inputs_base, inputs_embeds = model_dict['tokenized_datasets']['train']
    # print(inputs_base['input_ids'].shape, inputs_base['token_type_ids'].shape, inputs_base['attention_mask'].shape)
    # print(inputs_embeds['input_ids'].shape, inputs_embeds['token_type_ids'].shape, inputs_embeds['attention_mask'].shape)

    print("predicting baseline...")
    baseline_outputs = baseline_model(**inputs_base)
    baseline_last_hidden = baseline_outputs.last_hidden_state

    # print("predicticing embedding...")
    embed_outputs = embed_model(**inputs_embeds)
    embed_last_hidden = embed_outputs.last_hidden_state

    print(baseline_last_hidden.shape)
    print(embed_last_hidden.shape)

    inputs = torch.concat((baseline_last_hidden, embed_last_hidden), 1)
    torch.save(inputs, 'embed.pt')


    print("predicting baseline...")
    eval_baseline_outputs = baseline_model(**eval_inputs_base)
    eval_baseline_last_hidden = eval_baseline_outputs.last_hidden_state

    # print("predicticing embedding...")
    eval_embed_outputs = embed_model(**eval_inputs_embeds)
    eval_embed_last_hidden = eval_embed_outputs.last_hidden_state

    # print(baseline_last_hidden.shape)
    # print(embed_last_hidden.shape)

    eval_inputs = torch.concat((eval_baseline_last_hidden, eval_embed_last_hidden), 1)
    print(eval_inputs.shape)
    torch.save(eval_inputs, 'eval_embed.pt')
    # model = DenseModel(inputs.shape[2], 2)
    #
    # loss = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #
    # running_loss = 0.0
    # for epoch in tqdm(range(args.epochs)):
    #
    #     # zero the parameter gradients
    #     optimizer.zero_grad()
    #
    #     # forward + backward + optimize
    #     outputs = model(inputs)
    #     l = loss(outputs, labels)
    #     l.backward()
    #     optimizer.step()
    #
    #     # print statistics
    #     running_loss += l.item()
    #     # f1 = compute_metrics(outputs, lable_1d)
    #     # print("f1:", f1, 'loss:', loss.item())


def main():
    model_name = args.model
    assert model_name in set(MODEL_INFO.keys()), "Invalid model name"  # make sure it is a valid model name
    model_dict = MODEL_INFO[model_name]
    if model_name == 'full_model':
        train_full_model(model_dict)
        # model_dict['tokenizer'] = []
        # model_dict['model'] = []
        # for base_model in model_dict['base_model']:
        #     model_dict['tokenizer'].append(AutoTokenizer.from_pretrained(base_model))
        #     model_dict['model'].append(BertModel.from_pretrained(base_model)
    else:

        model_dict['tokenizer'] = BertTokenizer.from_pretrained(model_dict['base_model'])

        if model_name == 'baseline':
            model_dict['model'] = AutoModelForNextSentencePrediction.from_pretrained(model_dict['base_model'])
        elif model_name == 'derived_embed':
            model_dict['model'] = BertForSequenceClassification.from_pretrained(model_dict['base_model'])

        filedir = model_dict['model_dir'] + '/'

        if model_name == 'baseline':
            model_dict['dataset'] = load_dataset('csv', data_files={'train': DATAPATH + filedir + 'train.csv',
                                                                    'dev': DATAPATH + filedir + 'dev.csv',
                                                                    'test': DATAPATH + filedir + 'test.csv'})
        elif model_name == 'derived_embed':
            model_dict['dataset'] = load_dataset('csv', data_files=DATAPATH + filedir + 'embedding_train.csv')
        model_dict['tokenized_datasets'] = {}
        for name, data in model_dict['dataset'].items():
            model_dict['tokenized_datasets'][name] = make_datasets((name, data), model_dict['tokenizer'], args.max_len,
                                                                   model_name)
        if model_name == 'full_model':
            train_full_model(model_dict)
        else:
            train_model_part(model_dict)


if __name__ == '__main__':
    main()
