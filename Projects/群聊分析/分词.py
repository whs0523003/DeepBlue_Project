import sys
import jieba
from 数据读取和处理 import word_list


def stop_word():
    # 导入停用词列表
    stopwords = [line.strip() for line in open('./data/百度停用词表.txt', 'r', encoding='utf8').readlines()]
    # 除停用出外还需要移除的标点符号
    list_remove = [line.strip() for line in open('./data/标点符号数字停用表.txt', 'r', encoding='utf8').readlines()]
    # 空格需要特殊处理
    list_special = [' ']
    # 所有停用词合一
    all_stopwords = stopwords + list_remove + list_special
    return all_stopwords
all_stopwords = stop_word()


# jieba分词，过滤停用词
# 该分词用于统计词频
def jieba_split_count(word_list, num):
    # 把列表合成一个字符串
    word_str = ','.join(word_list)
    # 对字符串进行分词
    words = jieba.lcut(word_str)

    # 判断分词是否为停用词，是则丢弃，最后生成一个字典。
    list2 = []
    for word in words:
        if word not in all_stopwords:
            list2.append(word)

    hash = {}
    for i in list2:
        if i not in hash:
            hash[i] = 1
        else:
            hash[i] += 1

    # 对字典按照值排序
    # 对字典对象排序，按照字典的第2个值（即values）排序，返回一个元组。
    a = sorted(hash.items(), key=lambda x: x[1], reverse=True)

    # 返回前num个结果
    return a[:num]


# jieba分词，过滤停用词
# 该分词用于计算新文本和原始文本的相似度
def jieba_split_cal(word_list, new):
    # 用于存放被过滤后的句子
    list_all = []
    # 用于存放原句
    list_origin = []
    # 注：那些生成后的语料为空的句子会被删除，list_all和list_origin长度一样

    for idx, sentence in enumerate(word_list):
        words = ','.join(jieba.lcut(sentence)).split(',')
        filtered = ','.join([word for word in words if word not in all_stopwords])
        if len(filtered) >= 1:
            list_all.append(filtered)
            list_origin.append(word_list[idx])

    new_sentence = ','.join(jieba.lcut(new)).split(',')
    new_filter = [val for val in new_sentence if val not in all_stopwords]

    # 检测提问的语料库是否有效
    try:
        if len(new_filter) >= 1:
            pass
    except:
        print('提问的句子太短，无法生成有效语料')

    # 被过滤后的句子,原句，过滤后的输入
    return list_all, list_origin, new_filter



