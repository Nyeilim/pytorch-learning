from data_loader.load_fn import *
from torch import torch
from torch.nn import functional as F


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size  # 这个长度和独热编码的长度相同

    # 正态分布初始化参数，均值 0 方差 0.01
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数，num_hiddens
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 一开始没有隐变量，使用全 0 置位
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)，和上面 inputs 的形状对应
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q  # 过多层感知机，最后出来的结果 Y 是个矩阵
        outputs.append(Y)
    # outputs 是个列表 (nums_step, batch_size, len(vocab))，需要上下拼接成二维张量 (nums_step x batch_size, len(vocab))
    # 然后在和 H 左右拼接成元组，这里拼接是为了方便封装
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:

    def __init__(self, vocab_size, num_hiddens, device):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)

    # Callable 方法调用前向传播函数
    def __call__(self, X, state):
        # 颠倒 X，使之能够获得（时间步数，批量大小，词表大小）的 X，最后个维度就是独热编码
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return rnn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return init_rnn_state(batch_size, self.num_hiddens, device)
