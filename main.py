"""Main script for UDAEE."""

import param
from train import pretrain, evaluate
from model import (BertEncoder, DistilBertEncoder, DistilRobertaEncoder,
                   BertClassifier, Discriminator, RobertaEncoder, RobertaClassifier, EarlyBertEncoder, EarlyBertClassifier, EarlyRoBertaEncoder, EarlyRoBertaClassifier)
from utils import XML2Array, CSV2Array, convert_examples_to_features, \
    roberta_convert_examples_to_features, get_data_loader, init_model
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, RobertaTokenizer
import torch
import os
import random
import argparse
import pandas as pd
import csv


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="books",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb", "SST2", "QNLI", "SNLI"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="dvd",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb", "SST2", "QNLI", "SNLI"],
                        help="Specify tgt dataset")

    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/classifier')

    parser.add_argument('--adapt', default=False, action='store_true',
                        help='Force to adapt target encoder')

    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="exit_bert",
                        choices=["bert", "distilbert", "roberta", "distilroberta", "exit_bert", "early_roberta"],
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Specify adversarial weight")

    parser.add_argument('--beta', type=float, default=1.0,
                        help="Specify KD loss weight")

    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")

    parser.add_argument('--batch_size', type=int, default=16,
                        help="Specify batch size")

    parser.add_argument('--pre_epochs', type=int, default = 2,
                        help="Specify the number of epochs for pretrain")

    parser.add_argument('--pre_log_step', type=int, default=1,
                        help="Specify log step size for pretrain")

    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=2,
                        help="Specify log step size for adaptation")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_arguments()
    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seed: " + str(args.seed))
    print("train_seed: " + str(args.train_seed))
    print("model_type: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("pre_epochs: " + str(args.pre_epochs))
    print("num_epochs: " + str(args.num_epochs))
    print("AD weight: " + str(args.alpha))
    print("KD weight: " + str(args.beta))
    print("temperature: " + str(args.temperature))
    set_seed(args.train_seed)

    if args.model in ['roberta', 'distilroberta', 'early_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # preprocess data
    print("=== Processing datasets ===")
    if args.src in ['imdb']:
        src_x, src_y = CSV2Array(param.imdb_train_path)
        src_test_x, src_test_y = CSV2Array(param.imdb_test_path)
    elif args.src in ['SST2']:
        src_x, src_y = CSV2Array(param.sst_train_path)
        src_test_x, src_test_y = CSV2Array(param.sst_test_path)
    elif args.src in ['QNLI']:
        rte_path = param.qnli_train_path

        try:
            rte_data = pd.read_csv(rte_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            rte_data = pd.read_csv(rte_path, sep='\t', on_bad_lines='skip')

        src_x = (rte_data['question'] + " [SEP] " + rte_data['sentence']).tolist()
        src_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

        #################################################################################

        rte_path = param.qnli_test_path

        try:
            rte_data = pd.read_csv(rte_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            rte_data = pd.read_csv(rte_path, sep='\t', on_bad_lines='skip')

        src_test_x = (rte_data['question'] + " [SEP] " + rte_data['sentence']).tolist()
        src_test_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()



    elif args.src in ['SNLI']:
      snli_path = param.snli_train_path

      try:
          snli_data = pd.read_csv(snli_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
      except pd.errors.ParserError:
          snli_data = pd.read_csv(snli_path, sep='\t', on_bad_lines='skip')

      # Construct input pairs
      snli_data['sentence1'] = snli_data['sentence1'].fillna("").astype(str)
      snli_data['sentence2'] = snli_data['sentence2'].fillna("").astype(str)
      src_x = (snli_data['sentence1'] + " [SEP] " + snli_data['sentence2']).tolist()

      # Convert labels to numeric values
      label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
      src_y = snli_data['gold_label'].map(label_map).dropna().astype(int).tolist()

      #################################################################################

      snli_path = param.snli_test_path

      try:
          snli_data = pd.read_csv(snli_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
      except pd.errors.ParserError:
          snli_data = pd.read_csv(snli_path, sep='\t', on_bad_lines='skip')
      snli_data['sentence1'] = snli_data['sentence1'].fillna("").astype(str)
      snli_data['sentence2'] = snli_data['sentence2'].fillna("").astype(str)
      src_test_x = (snli_data['sentence1'] + " [SEP] " + snli_data['sentence2']).tolist()
      src_test_y = snli_data['gold_label'].map(label_map).dropna().astype(int).tolist()

    else:
        src_x, src_y = XML2Array(os.path.join('data', args.src, 'negative.review'),
                               os.path.join('data', args.src, 'positive.review'))

    # src_x, src_test_x, src_y, src_test_y = train_test_split(src_x, src_y,
    #                                                         test_size=0.25,
    #                                                         stratify=src_y,
    #                                                         random_state=args.seed)

    if args.src in ['imdb', 'SST2']:
        pass#tgt_x, tgt_y = CSV2Array(os.path.join('data', args.tgt, args.tgt + '.csv'))
    else:
      pass
        # tgt_x, tgt_y = XML2Array(os.path.join('data', args.tgt, 'negative.review'),
        #                          os.path.join('data', args.tgt, 'positive.review'))

    # tgt_train_x, tgt_test_y, tgt_train_y, tgt_test_y = train_test_split(tgt_x, tgt_y,
    #                                                                     test_size=0.2,
    #                                                                     stratify=tgt_y,
    #                                                                     random_state=args.seed)




    if args.model in ['roberta', 'distilroberta', 'early_roberta']:
        src_features = roberta_convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = roberta_convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        # tgt_features = roberta_convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
        # tgt_train_features = roberta_convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)
    else:
        src_features = convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        # tgt_features = convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
        # tgt_train_features = convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)

    # load dataset

    src_data_loader = get_data_loader(src_features, args.batch_size)
    # src_val_data_loader = get_data_loader(src_features, 1)
    src_data_eval_loader = get_data_loader(src_test_features, 1)
    # tgt_data_train_loader = get_data_loader(tgt_train_features, args.batch_size)
    # # tgt_data_train_val_loader = get_data_loader(tgt_train_features, 1)
    # tgt_data_all_v_loader = get_data_loader(tgt_features, args.batch_size)
    # tgt_data_all_loader = get_data_loader(tgt_features, 1)

    # load models
    if args.model == 'bert':
        src_encoder = BertEncoder()
        tgt_encoder = BertEncoder()
        src_classifier = BertClassifier()
    elif args.model == 'distilbert':
        src_encoder = DistilBertEncoder()
        tgt_encoder = DistilBertEncoder()
        src_classifier = BertClassifier()
    elif args.model == 'roberta':
        src_encoder = RobertaEncoder()
        tgt_encoder = RobertaEncoder()
        src_classifier = RobertaClassifier()
    elif args.model == 'exit_bert':
        src_encoder = EarlyBertEncoder()
        tgt_encoder = EarlyBertEncoder()
        src_classifier = EarlyBertClassifier()
    elif args.model == 'early_roberta':
        src_encoder = EarlyRoBertaEncoder()
        tgt_encoder = EarlyRoBertaEncoder()
        src_classifier = EarlyRoBertaClassifier()
    else:
        src_encoder = DistilRobertaEncoder()
        tgt_encoder = DistilRobertaEncoder()
        src_classifier = RobertaClassifier()
    discriminator = Discriminator()

    if args.load:
        src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path)
        src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path)
        tgt_encoder = init_model(args, tgt_encoder, restore=param.tgt_encoder_path)
        discriminator = init_model(args, discriminator, restore=param.d_model_path)
    else:
        src_encoder = init_model(args, src_encoder)
        src_classifier = init_model(args, src_classifier)
        tgt_encoder = init_model(args, tgt_encoder)
        discriminator = init_model(args, discriminator)

    # train source model
    print("=== Training classifier for source domain ===")
    if args.pretrain:
        src_encoder, src_classifier = pretrain(
            args, src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    print("On source validation")
    # evaluate(src_encoder, src_classifier, src_val_data_loader)
    print("source test dataset")
    _, df = evaluate(src_encoder, src_classifier, src_data_eval_loader)
    df.to_csv(param.dataframe_save_path, index = False)
    print("CSV saved succesfully, now proceed with simultaion by excuting the command (python simulation.py)")
    # evaluate(src_encoder, src_classifier, tgt_data_all_v_loader)

    # for params in src_encoder.parameters():
    #     params.requires_grad = False

    # for params in src_classifier.parameters():
    #     params.requires_grad = False

    # # train target encoder by GAN
    # print("=== Training encoder for target domain ===")
    # if args.adapt:
    #     tgt_encoder.load_state_dict(src_encoder.state_dict())
    #     tgt_encoder = adapt(args, src_encoder, tgt_encoder, discriminator,
    #                         src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_all_loader)

    # # eval target encoder on lambda0.1 set of target dataset
    # print("=== Evaluating classifier for encoded target domain ===")
    # print(">>> source only <<<")
    # evaluate(src_encoder, src_classifier, tgt_data_all_loader)
    # print(">>> domain adaption <<<")
    # evaluate(tgt_encoder, src_classifier, tgt_data_all_loader)


if __name__ == '__main__':
    main()
