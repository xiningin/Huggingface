import csv
import random
import torch
from transformers import BertTokenizer, BertModel

def mask_sentence(sentence , threshold=0.7):
    #将句子分割程单词
    words = sentence.split()
    label = ""
    #对于每个单词，使用随机数和阈值进行比较
    for i , word in enumerate(words):
        if random.random() > threshold:
            words[i] = '[MASK]'
            label = word
            #生成一个MASK就差不多了
            break

    return ' '.join(words) , label


if __name__ == "__main__":
    """
        检验一下mask_sentence生成掩码函数
    """
    # random.seed(416)
    # sentence = "The quick brown fox jumps over the lazy dog."
    # print(mask_sentence(sentence=sentence , threshold=0.7))

    """
        看一下bert的输出有哪些
    """
    # bertModel = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True, output_attentions=True)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # text = '让我们来看一下bert的输出都有哪些'
    # input_ids = torch.tensor([tokenizer.encode(text)]).long()
    # outputs = bertModel(input_ids)
    # print(outputs.keys())
    # print(f"pooler_output: {outputs['pooler_output'].shape}")

    """
        将数据生成含有掩码的数据
    """
    mask_sentence_data = []
    with open('sst2_shuffled.tsv' , 'r' , encoding="utf-8") as file:
        for sample in file.readlines():
            polar , sent = sample.strip().split("\t")
            mask_sentence_data.append(mask_sentence(sent))

    # for sample in mask_sentence_data:
    #     print(f"sent: {sample[0]} |||| label: {sample[1]}")
    # mask_sentence_data里面存好了sent与label






