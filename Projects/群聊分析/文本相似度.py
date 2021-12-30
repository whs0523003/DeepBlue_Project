import re
import sys
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from 数据读取和处理 import word_list
from 分词 import jieba_split_cal
from 文本向量化 import calculation


# 获取前topN个文本
def get_answer(topN):
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index
    idxs = np.argsort(cosdistances)[::-1][:topN]
    # 根据索引取值
    topN_documents = [word_list[idx] for idx in idxs]
    print('输入一句话：', new)
    print('可能的回答是:', topN_documents)


if __name__ == '__main__':
    cosdistances, new = calculation()
    get_answer(10)


