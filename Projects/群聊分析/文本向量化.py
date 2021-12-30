import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from 数据读取和处理 import word_list
from 分词 import jieba_split_cal
from gensim.models import Word2Vec

# 新的一句话
new = '什么'

# 计算tfidf
def tfidf():
    # 被过滤后的句子,原句，过滤后的输入
    list_all, list_origin, new_filter = jieba_split_cal(word_list, new)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(list_all).toarray()
    tfidf_new = vectorizer.transform(new_filter).toarray()[0]
    return tfidf, tfidf_new, new


# 计算2个向量的余弦相似度
def cosinsim(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    result = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return result


# 计算库里倒数第一条句子和前边所有句子的相似度
def calculation():
    result, result_new, new = tfidf()
    cosdistances = [cosinsim(sentence, result_new) for sentence in result]
    return cosdistances, new


def word2vec():
    # 引入数据集
    raw_sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]
    # 切分词汇,变为list of list格式
    sentences = [s.encode('utf-8').split() for s in raw_sentences]

    # 构建模型
    model = Word2Vec(sentences, min_count=1)
