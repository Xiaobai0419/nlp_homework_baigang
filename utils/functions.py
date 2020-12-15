import collections
import math
import os
import random
import sys
import tarfile
import time
import zipfile

from IPython import display
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
import numpy as np


'''
读取数据集并统计生成字典，将数据集处理为索引数组
'''
def load_data_jay_lyrics():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]  # 原始歌词，按歌词顺序返回其词典索引序列
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


'''
相邻采样
'''
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size #”长样本“长度，一共batch_size个这样的”长样本“
    indices = corpus_indices[0 : batch_size * batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps  #”长样本“包含歌词段数,等价于小批量数
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i : i + num_steps] #每行都取，各行取的组成一批，批之间对应每行歌词连续，所以可以让上一批隐藏层最后输出作为本批次隐藏层初始输入
        Y = indices[:, i + 1 : i + num_steps + 1]
        yield X, Y


'''
随机采样
'''
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a random order from sequential data."""
    num_examples = (len(corpus_indices) - 1) // num_steps #总歌词段数，每段是一个时间步长度
    epoch_size = num_examples // batch_size  #用于训练的小批量数
    example_indices = list(range(num_examples))  #为每段歌词编号
    random.shuffle(example_indices)  #每个输入是时间步长度的歌词段，是将前后有序的所有歌词分段，赋予序号后，随机打乱顺序

    def _data(pos):
        return corpus_indices[pos : pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i : i + batch_size] #每批所含歌词段序号，是随机乱序
        X = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx) #采每个歌词段的歌词字典索引，用于转为one-hot向量
        Y = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx) #对应歌词下一个字序列
        yield X, Y


'''
nd.one_hot(nd.array([0, 2]), vocab_size)

[[1. 0. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]]
<NDArray 2x1027 @cpu(0)>
'''
def to_onehot(X, size):
    """Represent inputs with one-hot encoding."""
    return [nd.one_hot(x, size) for x in X.T]


'''
裁剪梯度
'''
def grad_clipping(params, theta, ctx):
    """Clip the gradient."""
    if theta is not None:
        norm = nd.array([0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm


'''
小批量随机梯度下降
'''
def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
