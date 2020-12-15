import utils as us
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time


'''
读取歌词数据集
'''
(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = us.load_data_jay_lyrics()


'''
初始化模型参数
'''
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = None
def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # 附上梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


'''
初始化隐藏状态
'''
def init_rnn_state(batch_size, num_hiddens, ctx):  # 返回由一个形状为(批量大小, 隐藏单元个数)的值为0的NDArray组成的元组
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )


'''
计算一个时间步（歌词段）中每个预测字的隐藏状态和输出
'''
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps（时间步数）个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  #H初始状态
    outputs = []
    for X in inputs:  #每批单个字，批量输入
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)  # 更新H,每个H都是本次输入和上次H相加
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)  # 返回时间步矩阵，和最后一个H


'''
定义预测函数 基于前缀prefix（含有数个字符的字符串）来预测接下来的num_chars个字符
'''
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx) #1批
    output = [char_to_idx[prefix[0]]]  # 第一个字符的索引
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = us.to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)  # 一批，每批仅一个字，即单行单时间步，没有词间隐藏层作输入的一部分，因为是用来预测下一个字，而不是训练
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)  # 预测时是用训练好的参数进行运算，不再用上一个时间步的隐藏状态，所以不需要X是歌词段，而只是单个字符，返回的预测集合也仅单个字符，和一个隐藏状态
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])  # 有原始歌词就用原始歌词预测，否则用预测出的歌词预测下一个字
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))  # one-hot分量中最大值作为最优预测的索引，对应一个预测字
    return ''.join([idx_to_char[i] for i in output])  # 输出包括前缀在内的所有字符，即只预测前缀后面的字符并迭代作为输入来预测下一个，前缀的预测舍掉


'''
定义模型训练函数
'''
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = us.data_iter_random
    else:
        data_iter_fn = us.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):  #单纯的训练轮数，与训练数据集分成多少批无关
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)  # 用所有训练数据产生多批歌词用于小批量随机梯度下降
        for X, Y in data_iter:  # 每一批数据产生一组歌词段索引序列X（每个序列都作为RNN模型的一个多时间步输入），和一组对应的歌词下一个字标签序列Y
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                '''
                当多个相邻小批量通过传递隐藏状态串联起来时，模型参数的梯度计算将依赖所有串联起来的小批量序列。同一迭代周期中(epoch)，随着迭代次数的增加，梯度的计算开销会越来越大。 为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列，我们可以在每次读取小批量前将隐藏状态从计算图中分离出来
                批之间单行（歌词段）语义连续，用于利用RNN模型在批之间传递隐藏状态，产生语义连贯加强的训练效果
                '''
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = us.to_onehot(X, vocab_size)  # 单批各歌词段的字符都转为one-hot
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 连结之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,)) # Y,X都是转置，处理，然后按行序并为一列，因为Y,X原来就是对应的（歌词序列对应歌词下一次标签序列），所以这里也是一一对应的，用于计算交叉熵损失
                # 使用交叉熵损失计算平均分类误差 每个one-hot向量（对应一个字）计算交叉熵再求和取平均
                l = loss(outputs, y).mean()
            l.backward()
            us.grad_clipping(params, clipping_theta, ctx)  # 裁剪梯度
            # 每个小批量的所有歌词输出结果与对应标签计算交叉熵求和，并求了均值，这里对此采用梯度下降
            us.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.asscalar() * y.size  #所有批总损失
            n += y.size  # 所有批总字数

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))  # 计算一个平均总损失，这里做了指数运算
            for prefix in prefixes:  #每pred_period轮，每轮用所有批进行的完整训练后，打印损失，并打印预测出的歌词段
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2,
pred_period, pred_len, prefixes = 50, 50, ['爱像一阵风', '离开有你的季节']


'''
采用随机采样训练模型并创作歌词
'''
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)

'''
采用相邻采样训练模型并创作歌词

train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
'''