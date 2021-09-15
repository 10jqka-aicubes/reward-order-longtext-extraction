#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Filename:      train.py
#
# Description:
# Version:       1.0
# Created:       2021/06/17
# Company:       www.iwencai.com
#
# -------------------------------------------#
from reward_order_longtext_extraction.util.lstm_model import NerRNN
import pytorch_lightning as pl
import reward_order_longtext_extraction.util.data_util as data_util
from argparse import ArgumentParser
import reward_order_longtext_extraction.util.param_embedding_infer as infer

tag_map = {"PAD": 0, "B": 1, "I": 2, "E": 3, "O": 4}


def do_embedding(args):
    model_path = args.pretain_model
    file_list = [args.train_file_dir + "/train.json", args.train_file_dir + "/valid.json"]
    for file in file_list:
        infer.write_doc_vector(model_path, file)


def do_train(args):
    train_file = args.train_file_dir + "/train_vect.json"
    test_file = args.train_file_dir + "/valid_vect.json"

    embed, tags = data_util.get_emb_dataset(test_file)
    test_size = len(embed)
    dev_embed = embed[-int(test_size / 2) :]
    dev_tags = tags[-int(test_size / 2) :]

    train_embed, train_tags = data_util.get_emb_dataset(train_file)
    train_set = data_util.EmbeddingDataSet(train_embed, train_tags, tag_map)
    train_loader = data_util.padded_data_loader(data=train_set, batch_size=args.batch_size, workers=0)
    dev_set = data_util.EmbeddingDataSet(dev_embed, dev_tags, tag_map)
    dev_loader = data_util.padded_data_loader(data=dev_set, batch_size=1, workers=0)
    model = NerRNN(len(train_set.tag_map), **vars(args))
    # logger
    logger = pl.loggers.TensorBoardLogger(args.experiment_path, name=args.run_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="epoch_loss",
        save_top_k=1,
        mode="min"
    )
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_file_dir")
    parser.add_argument("--pretain_model")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser = NerRNN.add_model_specific_args(parser)
    args = parser.parse_args()
    # 0. 获得段向量embedding
    do_embedding(args)
    # 1. 训练
    do_train(args)
