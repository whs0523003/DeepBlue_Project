import re
import sys
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from wordcloud import WordCloud
from 数据读取和处理 import word_list
from 分词 import jieba_split_count

# 画图
def plot_bar(result):
    list_x = []
    list_y = []
    for i in result:
        list_x.append(i[0])
        list_y.append(i[1])

    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.bar(list_x, list_y)
    plt.show()
    return list_x


# 渲染词云
def render_cloud(result):
    wordcloud = WordCloud(font_path="./data/simsun.ttf").generate(result)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    # 停用词过滤，并按次数排序
    result = jieba_split_count(word_list, 50)
    # 画图
    word_result = plot_bar(result)
    # 画词云
    new_string = ','.join(word_result)
    render_cloud(new_string)