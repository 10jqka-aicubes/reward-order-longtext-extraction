#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Filename:      param_emb_infer.py
#
# Description:   输出段向量
# Version:       1.0
# Created:       2021/CURRENT_MOUNTH/05
# Author:        chaiyixuan@myhexin.com
# Company:       www.iwencai.com
#
# -------------------------------------------#
from sentence_transformers import SentenceTransformer
import torch
import os
import json
import time
from tqdm import tqdm


class ParamEncoder:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(str(model_path), device=self.device)

    def get_doc_vector(self, sentences, bs):
        """
        获取doc向量
        :param sentences 段落:
        :return list:
        """
        sentence_embeddings = self.model.encode(sentences, batch_size=bs)
        if sentence_embeddings is not None:
            return sentence_embeddings
        return []

    def get_doc_vect_batch(self, data):
        """
        @description :batch提速
        @param :
        @return :
        """
        sentences = []
        block_list = []
        for block in data["document"]:
            sentence = block["text"]
            sentences.append(sentence)
            block_list.append(block)
        vec_list = self.get_doc_vector(sentences, bs=10)
        assert len(data["document"]) == vec_list.shape[0]
        for block, vect in zip(data["document"], vec_list):

            block["text_vec"] = ",".join([str(vec) for vec in vect])
        yield data


def write_doc_vector(model_path, infer_file):
    """
    @description :获得每个doc的vector
    @param : model_path 模型路径， infer_file 需要提取向量的文件
    @return :
    """
    model = ParamEncoder(model_path)
    with open(infer_file, "r") as reader:
        data_list = []
        time_list = []
        # for origin_entry in tqdm(reader,  desc="reader"):
        for origin_entry in tqdm(reader):
            start = time.time()
            json_obj = json.loads(origin_entry)
            for obj in model.get_doc_vect_batch(json_obj):
                data_list.append(json.dumps(obj, ensure_ascii=False))
            end = time.time()
            time_list.append(end - start)
        # get_distribution(time_list)
    tail_name = ".json"
    file_list = infer_file.split(tail_name)
    out_file = file_list[0] + "_vect" + tail_name
    # print("out_file",out_file)
    with open(out_file, "w", encoding="utf-8") as writer:
        for line in data_list:
            writer.write(line + "\n")
        print("done")


if __name__ == "__main__":
    prject_path = os.getcwd()
    model_path = prject_path + "/block_tag/model/roberta_zh"
    # infer_file = "/root/project/data/cwb/content_long_sszc_train.json"
    infer_file = "/root/project/data/cwb/content_sszc_test_transd.json"
    write_doc_vector(model_path, infer_file)
