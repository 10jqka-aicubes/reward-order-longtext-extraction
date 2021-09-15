def get_pos(path, tag_map):
    begin_tag = tag_map.get("B")
    mid_tag = tag_map.get("I")
    end_tag = tag_map.get("E")

    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag and last_tag in [begin_tag]:
            tags.append([begin, begin])
            begin = -1

        last_tag = tag
    return tags


def get_f1(pre_path, tar_path, tag_map):
    origin = 0
    found = 0
    right = 0
    for batch in zip(tar_path, pre_path):
        tar, pred = batch
        tar_pos = get_pos(tar, tag_map)
        pred_pos = get_pos(pred, tag_map)

        origin += len(tar_pos)
        found += len(pred_pos)

        for p_tag in pred_pos:
            if p_tag in tar_pos:
                right += 1
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def get_point_pos(path, tag_map):
    """
    @description :获得双指针序列模型的位置
    @param :
    @return :
    """
    begin_tag = tag_map.get("B")
    end_tag = tag_map.get("E")
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    is_open = False  # 是否有B但是没有E
    tag_single_temp = []
    for index, tag in enumerate(path):
        # 1.1如果tag为B 且之前没出现过tag_temp
        if tag == begin_tag and len(tag_single_temp) == 0:
            begin = index
            is_open = True
            tag_single_temp = [begin, begin]
        # 1.2 如果tag为B tag_single_temp,则将tag_temp加入到tags中
        elif tag == begin_tag and len(tag_single_temp) != 0:
            begin = index
            is_open = True
            tags.append(tag_single_temp)
            tag_single_temp = [begin, begin]
        # 2. 如果现在的tag == end，tag 且之前有B
        elif tag == end_tag and is_open == True and begin_tag > -1:
            end = index
            tags.append([begin, end])
            is_open = False
            tag_single_temp = []
        last_tag = tag
    # 循环结束 还有 tag_single_temp，则加到tags
    if len(tag_single_temp) != 0:
        tags.append(tag_single_temp)
        tag_single_temp = []
    return tags


def get_point_f1(pre_path, tar_path, tag_map):
    """
    @description :获得双指针序列模型的f1值
    @param :
    @return :
    """
    origin = 0
    found = 0
    right = 0
    for batch in zip(tar_path, pre_path):
        tar, pred = batch
        tar_pos = get_point_pos(tar, tag_map)
        pred_pos = get_point_pos(pred, tag_map)

        origin += len(tar_pos)
        found += len(pred_pos)

        for p_tag in pred_pos:
            if p_tag in tar_pos:
                right += 1
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


if __name__ == "__main__":
    is_point = True
    if is_point:
        path = [3, 2, 3, 1, 3, 2, 1, 1, 1, 3, 3, 1, 2]
        tag_map = {"B": 1, "E": 2, "O": 3}
        p = get_point_pos(path, tag_map)
        print(p)
        pred_path = [[3, 3, 3, 1, 3, 2, 0, 0, 1, 3, 3, 1, 3]]
        tar_path = [[3, 3, 3, 1, 3, 2, 0, 0, 1, 3, 3, 1, 2]]
        f1, p, r = get_point_f1(pred_path, tar_path, tag_map)
        print(f1, p, r)
    else:
        path = [0, 0, 0, 1, 2, 3, 0, 0, 1, 0, 0, 1, 3]
        tag_map = {"B": 1, "I": 2, "E": 3, "O": 0}
        p = get_pos(path, tag_map)
        print(p)
        pred_path = [[0, 0, 0, 1, 2, 3, 0, 0, 1, 0, 0, 1, 3]]
        tar_path = [[0, 0, 0, 1, 2, 3, 0, 0, 1, 0, 0, 0, 0]]
        f1, p, r = get_f1(pred_path, tar_path, tag_map)
        print(f1, p, r)

def trans_para_to_char(obj, para_pos):
    doc_len_list = [len(doc["text"]) for doc in obj["document"]]
    new_pos = []
    for po in para_pos:
        start_block = po[0]
        end_block = po[1]
        if start_block != 0:
            start = 0 + sum(doc_len_list[:start_block])
        else:    
            start = 0
        end = sum(doc_len_list[:end_block+1])-1
        new_pos.append([start, end])
    return new_pos