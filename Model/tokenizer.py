import sys
from transformers import AutoTokenizer

sys.path.append("..")

import argparse
import os


# 这里的max_len需要自己设置（超参数），pt_model是hugging_face上面的预训练模型地址
def deepscc_tokenizer(data, max_len=150, pt_model="NTUYG/DeepSCC-RoBERTa"):
    # model_pretained_name = "NTUYG/DeepSCC-RoBERTa"  # 'bert-base-uncased'
    # model_pretained_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pt_model)     #
    if max_len == 0:
        '''
            batch_encode_plus类似于padding，将多个输入序列 同时编码为 固定长度的向量
            
            主要作用：
                    对输入的序列 批次进行 编码， 将这些序列 转换为 模型能够处理的格式
        '''
        tokenized = tokenizer.batch_encode_plus(
            data,
            # max_length = max_len,
            pad_to_max_length=True,
            truncation=True
        )
    else:
        tokenized = tokenizer.batch_encode_plus(
            data,
            max_length = max_len,
            pad_to_max_length = True,
            truncation = True
        )

    return tokenized, tokenizer.vocab_size
