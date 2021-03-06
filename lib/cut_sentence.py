"""
cut the sentence into words
"""
import jieba
import string
import config as cfg
from lib import stopwords
import logging


# 关闭jieba log输出
jieba.setLogLevel(logging.INFO)

# 准备英文字符
letters = string.ascii_lowercase + "+"


def cut_sentence_by_word(sentence):
    """
    实现中英文分词
    :param sentence:
    :return:
    """
    # python和c++哪个难？--> [python, 和, c++, 哪, 个, 难, ？]
    result = []
    temp = ""
    for word in sentence:
        # 把英文单词进行拼接
        if word.lower() in letters:
            temp += word
        else:
            if temp != "":  # 出现中文，把英文添加到结果中
                result.append(temp.lower())
                temp = ""

            if len(word.strip()) > 0:
                result.append(word.strip())

    if temp != "":  # 判断最后的字符是否为英文
        result.append(temp.lower())

    return result


def cut(sentence: str, by_word=False, use_stopwords=False, with_sg=False):
    """
    :param sentence: str should be cleaned sentence, without 'M: ' in front of the sentence
    :param by_word: cut a sentence by merely words like a,b,c,d..., '你','好'
    :param use_stopwords: ending with ends mark
    :param with_sg:
    :return:
    """

    # to make sure twice
    sentence = sentence.strip()

    if by_word:
        result = cut_sentence_by_word(sentence)
    else:
        result = jieba.lcut(sentence)
        # result = [(i.word, i.flag) for i in result if len(i.word.strip()) > 0]
        if with_sg:
            result = [i[0] for i in result if len(i[0].strip()) > 0]

    # if use_stopwords:
    #     result = [i for i in result if i not in stopwords()]

    return result
