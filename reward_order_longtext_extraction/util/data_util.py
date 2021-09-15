import reward_order_longtext_extraction.util.util as util
import json
import torch
import torch.utils.data as dt
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import re


def text_vec2vec(text_vec):

    e = text_vec.split(",")
    e = [float(ei) for ei in e]
    return e


def get_emb_dataset(file_name):
    """
    @description : 根据标注平台格式的file_name，获得输入特征和标签
    @param :file_name: 数据集地址
    @return : vect_list：段落特征的列表，维度是（数据总量，段落数，特征维度）；
    tag_all_list：段落对应的标签，维度是（数据总量，段落数）
    """
    data_list = util.read_file(file_name)
    vect_list = []  # all_num , block_len, emb_vec
    tag_all_list = []
    multi_num = 0
    single_num = 0
    no_num = 0
    len_list = []
    id = 0
    for data in data_list:
        if id == 42:
            print("id")
        id += 1
        obj = json.loads(data)
        block_list = []
        tag_list = []
        for block in obj["document"]:
            text_vec = block["text_vec"]
            block_list.append(text_vec2vec(text_vec))
        block_num = len(block_list)
        for qa in obj["qas"][0]:
            for answer in qa["answers"]:
                start_b = int(answer["start_block"])
                end_b = int(answer["end_block"])
                tag_list.append((start_b, end_b))
        tag_list.sort(key=lambda t: t[0])
        tag_name_list = ["O" for _ in range(block_num)]
        # is_start = False
        if len(tag_list) > 1:
            multi_num += 1
        elif len(tag_list) == 0:
            no_num += 1
        else:
            single_num += 1
        # for i in range(block_num): # 0,1,2,3,4,5
        #     for start_b, end_b in tag_list: # 0,2 3,5
        #         if start_b == end_b and end_b == i:
        #             tag_name_list[i] = 'B'
        #             continue
        #         if i == start_b and tag_name_list[i] == 'O':
        #             is_start = True
        #             tag_name_list[i] = 'B'

        #         elif i == end_b and tag_name_list[i] == 'O':
        #             is_start = False
        #             tag_name_list[i] = 'E'

        #         elif is_start and tag_name_list[i] == 'O' and end_b > i:
        #             tag_name_list[i] = 'I'
        # print("tag_name_list", tag_name_list)
        # 更换标注逻辑
        for start_b, end_b in tag_list:
            if start_b == end_b:
                tag_name_list[start_b] = "B"
            else:
                tag_name_list[start_b] = "B"
                for i in range(start_b + 1, end_b):
                    tag_name_list[i] = "I"
                tag_name_list[end_b] = "E"

        vect_list.append(block_list)
        tag_all_list.append(tag_name_list)
        len_list.append(len(tag_name_list))
        if "B" not in tag_name_list:
            print("no anser")
    # print(tag_all_list)
    util.get_sentence_len_distribution(len_list)
    # print(len_list)
    print("no_num ", no_num, "single_num ", single_num, "multi_num ", multi_num)

    return vect_list, tag_all_list


def get_title_emb_dataset(
    file_name,
):
    """
    @description : 根据正则，数字标题加入feature，根据标注平台格式的file_name，获得输入特征和标签
    @param :file_name: 数据集地址
    @return : vect_list：段落特征的列表，维度是（数据总量，段落数，特征维度）；
    tag_all_list：段落对应的标签，维度是（数据总量，段落数）
    """
    data_list = util.read_file(file_name)
    vect_list = []  # all_num , block_len, emb_vec
    tag_all_list = []
    multi_num = 0
    single_num = 0
    no_num = 0
    len_list = []
    id = 0
    for data in data_list:
        if id == 42:
            print("id")
        id += 1
        obj = json.loads(data)
        block_list = []
        tag_list = []
        for block in obj["document"]:

            text_vec = block["text_vec"]

            text_vec = text_vec2vec(text_vec)

            block_text = block["text"]

            pattern = r"^一、|^二、|^三、|^四、|^五、|^六、|^七、|^八、|^九、|^十(.?)、|^[(（]一|^[(（]二|^[(（]三|^[(（]四|^[(（]五|^[(（]六|^[(（]七|^[(（]八|^[(（]九|^[(（]十|一$|二$|三$|四$|五$|六$|七$|八$|九$|十$"
            flag = re.search(pattern, block_text)
            if flag:
                text_vec = [vec + 1 for vec in text_vec]
                print(block_text)
            block_list.append(text_vec)
        block_num = len(block_list)
        for qa in obj["qas"][0]:
            for answer in qa["answers"]:
                start_b = int(answer["start_block"])
                end_b = int(answer["end_block"])
                tag_list.append((start_b, end_b))
        tag_list.sort(key=lambda t: t[0])
        tag_name_list = ["O" for _ in range(block_num)]
        is_start = False
        if len(tag_list) > 1:
            multi_num += 1
        elif len(tag_list) == 0:
            no_num += 1
        else:
            single_num += 1
        for i in range(block_num):  # 0,1,2,3,4,5
            for start_b, end_b in tag_list:  # 0,2 3,5
                if start_b == end_b and end_b == i:
                    tag_name_list[i] = "B"
                    continue
                if i == start_b and tag_name_list[i] == "O":
                    is_start = True
                    tag_name_list[i] = "B"

                elif i == end_b and tag_name_list[i] == "O":
                    is_start = False
                    tag_name_list[i] = "E"

                elif is_start and tag_name_list[i] == "O" and end_b > i:
                    tag_name_list[i] = "I"
                # print("tag_name_list", tag_name_list)

        vect_list.append(block_list)
        tag_all_list.append(tag_name_list)
        len_list.append(len(tag_name_list))
        if "B" not in tag_name_list:
            print("no anser")
    print(tag_all_list)
    util.get_sentence_len_distribution(len_list)
    print(len_list)
    print("no_num ", no_num, "single_num ", single_num, "multi_num ", multi_num)

    return vect_list, tag_all_list


def get_point_emb_dataset(
    file_name,
):
    """
    @description : 根据标注平台格式的file_name，获得输入特征和标签
    @param :file_name: 数据集地址
    @return : vect_list：段落特征的列表，维度是（数据总量，段落数，特征维度）；
    tag_all_list：段落对应的标签，维度是（数据总量，段落数）
    """
    data_list = util.read_file(file_name)
    vect_list = []  # all_num , block_len, emb_vec
    tag_all_list = []
    multi_num = 0
    single_num = 0
    no_num = 0
    len_list = []
    for data in data_list:
        obj = json.loads(data)
        block_list = []
        tag_list = []
        for block in obj["document"]:
            text_vec = block["text_vec"]
            block_list.append(text_vec2vec(text_vec))
        block_num = len(block_list)
        for qa in obj["qas"][0]:
            for answer in qa["answers"]:
                start_b = int(answer["start_block"])
                end_b = int(answer["end_block"])

                tag_list.append((start_b, end_b))

        tag_name_list = ["O" for _ in range(block_num)]
        is_start = False
        if len(tag_list) > 1:
            multi_num += 1
        elif len(tag_list) == 0:
            no_num += 1
        else:
            single_num += 1
        for i in range(block_num):  # 0,1,2,3,4,5

            for start_b, end_b in tag_list:  # 0,2 3,5
                if start_b == end_b and end_b == i:
                    tag_name_list[i] = "B"
                    continue
                if i == start_b and tag_name_list[i] == "O":
                    is_start = True
                    tag_name_list[i] = "B"

                elif i == end_b and tag_name_list[i] == "O":
                    is_start = False
                    tag_name_list[i] = "E"

                # elif is_start and tag_name_list[i] == 'O' and end_b > i:
                #     tag_name_list[i] = 'I'
                # print("tag_name_list", tag_name_list)

        vect_list.append(block_list)
        tag_all_list.append(tag_name_list)
        len_list.append(len(tag_name_list))
        if "B" not in tag_name_list:
            print("no anser")
    print(tag_all_list)
    util.get_sentence_len_distribution(len_list)
    print(len_list)
    print("no_num ", no_num, "single_num ", single_num, "multi_num ", multi_num)

    return vect_list, tag_all_list


class EmbeddingDataSet(dt.Dataset):
    """
    @description : 返回Dataset对象，供dataloader使用
    @param :
    @return :
    """

    def __init__(self, embeddings, tags, tag_map):
        # self.__build_tags_vocab(tags)
        self.tag_map = tag_map
        self.x = [np.array(embedding, dtype="float32") for embedding in embeddings]
        self.y = [self.__replace_with_tag_index(tag) for tag in tags]

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])

    def __len__(self):
        return len(self.x)

    # def __build_tags_vocab(self, tags):
    #     tags = list(set([tag for tag_el in tags for tag in tag_el]))
    #     self.tag_vocab = defaultdict(int)
    #     self.tag_vocab['PAD'] = 0
    #     for token in tags:
    #         self.tag_vocab[token] = len(self.tag_vocab)

    def __replace_with_tag_index(self, tags):
        return np.array([self.tag_map[token] for token in tags], dtype="int64")


def padding(data):
    """
    @description :按照batch中最大的padding
    @param :
    @return :
    """
    x, y = zip(*data)
    x = pad_sequence(x, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return x, y


def padding_max_len(data):
    """
    @description :按照max_len padding
    @param :
    @return :
    """

    def pad_max_sequence(sequences, padding_value=0.0, max_len=420):
        # type: (List[Tensor], bool, float) -> Tensor
        # assuming trailing dimensions and type of all the Tensors
        # in sequences are same and fetching those from sequences[0]
        max_size = sequences[0].size()
        # print(max_size)
        trailing_dims = max_size[1:]
        # print("trailing_dims",trailing_dims)

        out_dims = (len(sequences), max_len) + trailing_dims

        # print("out_dims",out_dims)

        out_tensor = sequences[0].new_full(out_dims, padding_value)
        # print("out_tensor",out_tensor)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor

            out_tensor[i, :length, ...] = tensor[:max_len]

        return out_tensor

    x, y = zip(*data)
    x = pad_max_sequence(x)
    y = pad_max_sequence(y)
    mask = x.ne(0)
    return x, y, mask


def padded_data_loader(data, workers=0, batch_size=32):
    """
    @description :padded的意思是每个batch 按照最大句长padding
    @param :
    @return :
    """
    return dt.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, collate_fn=padding, num_workers=workers)


def single_data_loader(data, workers=1):
    """
    @description :batch_size 为1
    @param :
    @return :
    """
    return dt.DataLoader(dataset=data, batch_size=1, shuffle=True, num_workers=workers)


def max_len_data_loader(data, batch_size, workers=1, max_len=420):
    """
    @description :batch_size 为1
    @param :
    @return :
    """
    return dt.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True, collate_fn=padding_max_len, num_workers=workers
    )
