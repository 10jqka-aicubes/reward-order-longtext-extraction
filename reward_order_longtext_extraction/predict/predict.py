#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Filename:
#
# Description:
# Version:       1.0
# Company:       www.10jqka.com.cn
#
# -------------------------------------------#
import sys
sys.path.append("/home/jovyan/reward-order-longtext-extraction")
from reward_order_longtext_extraction.util.lstm_model import NerRNN
import reward_order_longtext_extraction.util.data_util as data_util
import reward_order_longtext_extraction.util.ner_util as ner_util
import reward_order_longtext_extraction.util.pl_util as pl_util
import reward_order_longtext_extraction.util.util as util
from argparse import ArgumentParser
import numpy as np
import torch
import json
from pathlib import Path
import reward_order_longtext_extraction.util.param_embedding_infer as infer


version = "version_0"
tag_map = {"PAD" : 0, "B" : 1, "I" : 2, "E" : 3, "O" : 4}


def do_predict(args):
    test_file = args.predict_file_dir + "/content_test.json"
    infer.write_doc_vector(args.pretain_model, test_file)
    test_file = args.predict_file_dir + "/content_test_vect.json"

    data_json_list = util.read_json_file(test_file)
    embed, tags = data_util.get_emb_dataset(test_file)
    model = pl_util.load_model_from_experiment(NerRNN, args.experiment_path + "/default/version_0/")
    model = model.to("cuda:0")

    found = 0
    origin = 0
    right_all = 0
    pref = Path(args.predict_result_file_dir)
    pref.mkdir(parents=True, exist_ok=True)
    with (pref / "predict.txt").open("w", encoding="utf-8") as f:
        for emd, tag, obj in zip(embed, tags, data_json_list):
            emd = torch.from_numpy(np.array(emd, dtype="float32")).to("cuda:0")
            emd = emd.unsqueeze(0)
            tag = np.array([tag_map[token] for token in tag], dtype="int64")

            outputs = model(emd)
            predicted = outputs.argmax(2)
            pred_pos = ner_util.get_pos(predicted[0], tag_map)
            truth_pos = ner_util.get_pos(tag, tag_map)
            right = 0
            for pp in pred_pos:
                if pp in truth_pos:
                    right += 1

            right_all += right
            origin += len(truth_pos)
            found += len(pred_pos)
            doc_text = ""
            for doc in obj["document"]:
                doc_text += doc["block_id"] + doc["text"] + "\n"
            new_pos = ner_util.trans_para_to_char(obj, pred_pos)
            f.write(json.dumps({"result": new_pos}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_path", type=str,default="/home/jovyan/model/")
    parser.add_argument("--predict_file_dir", type=str,default="/home/jovyan/data/sample_704")
    parser.add_argument("--predict_result_file_dir", type=str,default="/home/jovyan/data/predict_result/")
    parser.add_argument("--pretain_model", type=str,default="/read-only/common/pretrain_model/sentence_transformers/roberta_zh/")
    parser = NerRNN.add_model_specific_args(parser)
    args = parser.parse_args()
    do_predict(args)
