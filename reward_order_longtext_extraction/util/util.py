import json
import matplotlib.pyplot as plt


def remove_join(set_a, set_b):
    return set_a - set_b


def read_file(file_name):
    with open(file_name, "r", encoding="utf-8") as reader:
        data_list = []
        for line in reader:
            line = line.strip()
            data_list.append(line)
        return data_list


def read_json_file(file_name):
    with open(file_name, "r", encoding="utf-8") as reader:
        data_list = []
        for line in reader:
            # line=line.strip()
            obj = json.loads(line)
            data_list.append(obj)
        return data_list


def write_file(data_list, file_name, file_name_idx=None):
    """
    write_to_file
    @param
    file_name_idx: 为file_name后的数字区分符
    """
    if file_name_idx != None:
        name_list = file_name.split(".")
        file_name = "".join(name_list[:-1]) + "_" + str(file_name_idx) + "." + name_list[-1]
    with open(file_name, "w", encoding="utf-8") as writer:
        for data in data_list:
            data = data.strip()
            writer.write(data + "\n")
    print("Write to ", file_name, " done.")


def write_json_file(data_list, file_name, file_name_idx=None):
    """
    write_to_file
    @param
    file_name_idx: 为file_name后的数字区分符
    """
    if file_name_idx != None:
        name_list = file_name.split(".")
        file_name = "".join(name_list[:-1]) + "_" + str(file_name_idx) + "." + name_list[-1]
    with open(file_name, "w", encoding="utf-8") as writer:
        for data in data_list:
            data = json.dumps(data, ensure_ascii=False)
            data = data.strip()
            writer.write(data + "\n")
    print("Write to ", file_name, " done.")


"""

分k-flod
"""


def split_k_fold(data_list, k, file_name):
    # 　import random
    # 　random.shuffle(data_list)
    all_num = len(data_list)
    print("all_num", all_num)
    one_fold_num = int(all_num / k)
    splited_list = []
    for i in range(k):
        one_fold_list = data_list[:one_fold_num]
        data_list = data_list[one_fold_num:]
        splited_list.append(one_fold_list)

    for i, test_list in enumerate(splited_list):
        train_list = []
        for j, cur_list in enumerate(splited_list):
            if i == j:
                continue
            else:
                train_list.extend(splited_list[j])
        print("test_list", len(test_list))
        print("train_list", len(train_list))

        write_file(test_list, file_name + "_test_fold" + str(i) + ".json")
        write_file(train_list, file_name + "_train_fold" + str(i) + ".json")


"""
绘制句长分布图
"""


def draw_sent_len(data_list, max_len=None):
    def darw_hist(data_list):
        plt.hist(data_list, 50)
        plt.show()

    content_lens = []
    for data in data_list:
        length = len(data)
        if max_len:
            if length < max_len:
                content_lens.append(length)
        else:
            content_lens.append(length)

    plt.hist(content_lens, 50)
    plt.show()
    return plt


def split_train_test(data_list, file_name, split_radio=0.9):
    """
    分测试集训练集,写到file_name_train.后缀名,写到file_name_test.后缀名中
    """
    import random

    random.seed(6)
    random.shuffle(data_list)
    all_num = len(data_list)
    path_name, tail_name = file_name.split(".")
    write_file(data_list[: int(0.9 * all_num)], path_name + "_train." + tail_name)
    write_file(data_list[int(0.9 * all_num) :], path_name + "_test." + tail_name)
    print("all num ", all_num)
    print("write file done")


def transfer_file(origin_file, target_file):
    """
    将target_file 内容根据key复制到origin_file中去
    """
    origin_data_list = read_file(origin_file)
    target_data_list = read_file(target_file)
    json_list = []
    same_num = 0
    for idx, origin_data in enumerate(origin_data_list):
        origin_data = json.loads(origin_data)

        for target_data in target_data_list:
            target_data = json.loads(target_data)
            if origin_data["key"] == target_data["key"]:
                origin_data["qas"] = target_data["qas"]
                same_num += 1
        json_raw = json.dumps(origin_data, ensure_ascii=False)
        json_list.append(json_raw)
    path_name, tail_name = origin_file.split(".")
    file_name = path_name + "_transed." + tail_name
    print("key same num is ", same_num)
    write_file(json_list, file_name)


import matplotlib.pyplot as plt

# 引用numpy模块
import numpy as np

# from wordcloud import WordCloud
import jieba

"""
draw_sentence_len_distribution  获得句长分布统计图
"""


def draw_sentence_len_distribution(sentence_len_list):
    _ = plt.hist(sentence_len_list, bins=100)
    plt.xlabel("sentece length ")
    plt.ylabel("count num")
    plt.title("Distribution of sentece lengtht")

    plt.show()


def get_sentence_len_distribution(sentence_len_list):
    """
    @description :获得句长分布属性
    @param :
    @return :
    """
    len_array = np.array(sentence_len_list)
    # 求数组a的四分位数
    quarter = np.percentile(len_array, [25, 50, 75])
    max_num = np.max(len_array)
    min_num = np.min(len_array)
    mean = np.mean(len_array)
    print("**************** 统计信息：*****************")
    print("文本数据的最大句长:", max_num, ",最小句长:", min_num, ",平均值:", mean, ",小四分位数、中位数、大四分位数:", quarter)


"""
word_cloud： 词云绘制函数
parm: data_list 文本list   
                    e.g.["今天天气好","我不知道", ...]
"""
# def word_cloud(data_list):


#     # 结巴分词，生成字符串，wordcloud无法直接生成正确的中文词云
#     all_str=""
#     for line in data_list:
#         #print("line",line)
#         all_str+=line
#     cut_text = " ".join(jieba.cut(all_str))
#     #print("cut_text",cut_text)
#     #cut_text = cut_text.join(jieba.cut(data_list[1]))
#     #去除停用词
#     stop_words=["之后","以及","目前","包括","虽然","但是","表示" ,"今天", "由于", "昨天", "虽然", "一个","可以","必须","已经"]
#     with open("data/cn_stopwords.txt","r",encoding="utf-8") as reader:
#         for line in reader:
#             stop_words.append(line.strip())
#     sw = set(stop_words)

#     wordcloud = WordCloud(
#         # 设置字体，不然会出现口字乱码，文字的路径是电脑的字体一般路径，可以换成别的
#         font_path="C:/Windows/Fonts/simfang.ttf",
#         # 设置了背景，宽高
#         background_color="white", width=1800, height=1600,max_words = 100,stopwords=sw).generate(cut_text)

#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     plt.show()


"""
getSentenceCharacter: 获得文本统计信息

parm: data_list 文本list   
                    e.g.["今天天气好","我不知道", ...]

"""


def getSentenceCharacter(data_list):
    min_num = 200
    max_num = 0

    sentence_len_list = []
    max_len = 4018
    out_list = []
    for line in data_list:
        line_num = len(line)
        if line_num > max_len:
            out_list.append(1)
        else:
            out_list.append(0)

        sentence_len_list.append(line_num)
    print("out max_len prob =", np.mean(out_list) * 100, "%")
    len_array = np.array(sentence_len_list)
    # 求数组a的四分位数
    quarter = np.percentile(len_array, [25, 50, 75])
    max_num = np.max(len_array)
    min_num = np.min(len_array)
    mean = np.mean(len_array)
    print("**************** 统计信息：*****************")
    print("文本数据的最大句长:", max_num, ",最小句长:", min_num, ",平均值:", mean, ",小四分位数、中位数、大四分位数:", quarter)
    print("\n句长分布图：")
    draw_sentence_len_distribution(len_array)
    # print("\n词云图：")# word_cloud(data_list)
    return mean, quarter, max_num, min_num, len_array


