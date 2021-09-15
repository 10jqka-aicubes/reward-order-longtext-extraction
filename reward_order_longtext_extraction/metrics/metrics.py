#!/usr/bin/env python
# encoding:utf-8
# -------------------------------------------#
# Filename:      eval.py
#
# Description:      自动评估代码
#                   1.输出格式
#                    输出文件名: result.json
#                    文件格式说明： 预测段落位置列表
#                    {"result": [[10,20],[25,30]]}
#                    如果没有答案则输出
#                    {"result": []}
#                    2. 评估
#                    pred_file：预测文件
#                    truth_file：真实标签文件
#                    评估结果存在eval_result.txt中
# Version:       1.0
# Created:       2021/CURRENT_MOUNTH/29
# Author:        chaiyixuan@myhexin.com
# Company:       www.iwencai.com
#
# -------------------------------------------#
import json
from argparse import ArgumentParser


def get_f1(pred_spans_list, truth_spans_list, detail_file):
    """
    # Parameters
    pred_spans_list: 预测的段落位置列表，由开始段落id和结束段落id组成。
                            e.g. [[[25,37],[39,42]],[[0,3]]] 预测答案为文章1的25段到37段
                            和39段到42段, 文章2的0段到3段。
    truth_spans_list 真实的段落位置列表，由开始段落id和结束段落id组成。
                            e.g. [[[25,37],[39,42],[43,58]],[[0,3]]] 真实答案为25段到37段,
                            39段到42段和43段到58段, 文章2的0段到3段。
    """
    pred_num = 0
    truth_num = 0
    right_num = 0

    oup = open(detail_file, "w", encoding="utf-8")
    oup.write("pred\ttruth\tresult\n")
    for pred_spans, truth_spans in zip(pred_spans_list, truth_spans_list):
        is_match = False
        for pred_span in pred_spans:
            if pred_span in truth_spans:
                is_match = True
                right_num += 1
        line = "{}\t{}\t{}\n".format(
            json.dumps(pred_spans), 
            json.dumps(truth_spans),
            is_match
        )
        oup.write(line)
        pred_num += len(pred_spans)
        truth_num += len(truth_spans)
    recall = 0 if truth_num == 0 else (right_num / truth_num)
    precision = 0 if pred_num == 0 else (right_num / pred_num)
    f1 = 0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def read_json_file(file_name):
    with open(file_name, "r", encoding="utf-8") as reader:
        data_list = []
        for line in reader:
            # line=line.strip()
            obj = json.loads(line)
            data_list.append(obj)
        return data_list


def concat_result(json_list):
    result_list = []
    for obj in json_list:
        result_list.append(list(obj.values())[0])
    return result_list


def eval(args):
    pred_json_list = read_json_file(args.predict_result_file_dir + "/predict.txt")
    truth_json_list = read_json_file(args.groudtruth_file_dir + "/result.json")
    pred_span_list = concat_result(pred_json_list)
    truth_span_list = concat_result(truth_json_list)
    detail_file = args.result_detail_file
    f1, p, r = get_f1(pred_span_list, truth_span_list, detail_file)
    print("F1:", f1, "Precision:", p, "Recall:", r)
    with open(args.result_json_file, "w", encoding="utf-8") as writer:
        data_map = {"F1": f1, "Precision": p, "Recall": r}
        writer.write(json.dumps(data_map))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predict_result_file_dir", type=str,default = "/home/jovyan/data/predict_result/")
    parser.add_argument("--groudtruth_file_dir", type=str,default="/home/jovyan/data/sample_704/")
    parser.add_argument("--result_json_file", type=str,default="/home/jovyan/data/predict_result/result.json")
    parser.add_argument("--result_detail_file", type=str,default="/home/jovyan/data/predict_result/result.detail")
    args = parser.parse_args()
    eval(args)
