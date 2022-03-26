from tools.doccano_transformer import read_jsonl, NERDataset
import re


def get_word_list(s1):
    # 中文BIO格式分行：[\u4e00-\u9fa5]中文范围 ，把句子按字分开，中文按字分，英文按单词，数字按空格 数字也得全部隔开
    res = re.compile(
        r"([\u4e00-\u9fa5]|[\uFF00-\uFFEF]|[\u3000-\u303F]|[\u3200-\u32FF]|[\uFE10–\uFE1F]|[\u2018-\u2030]|\d)")
    p1 = res.split(s1)
    # print(p1)
    str1_list = []
    for s in p1:
        if res.split(s) is None:
            str1_list.append(s)
        else:
            ret = res.split(s)
            for ch in ret:
                str1_list.append(ch)
    list_words = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
    lists = []
    for list_word in list_words:
        splits = list_word.split()
        # print(split)
        for split in splits:
            lists.append(split)
    return lists


dataset = read_jsonl(filepath='origin.jsonl', dataset=NERDataset, encoding='utf-8')
dataset_processed = dataset.to_bio_format(tokenizer=get_word_list)
dataset_processed = list(dataset_processed)
total_num = len(dataset_processed)

with open("../data/train.txt", "w") as train:
    # 80% 用于 train
    for i, d in enumerate(dataset_processed):
        if i + 1 <= total_num * 0.8:
            train.write(d['data'])
        else:
            break

with open("../data/test.txt", "w") as test:
    # 20% 用于 test
    for i, d in enumerate(dataset_processed):
        if i + 1 > total_num * 0.8:
            test.write(d['data'])