import torch
from torch.nn import functional as F

from rnn import RNNModelScratch


def get_params(vocab_size, num_hiddens_1, num_hiddens_2, device):
    num_inputs = num_outputs = vocab_size  # 这个长度和独热编码的长度相同

    # 正态分布初始化参数，均值 0 方差 0.01
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数（下面那层），num_hiddens_1
    U = normal((num_inputs, num_hiddens_1))
    W = normal((num_hiddens_1, num_hiddens_1))
    s_1 = torch.zeros(num_hiddens_1, device=device)

    # 隐藏层参数（上面那层），num_hiddens_2
    V = normal((num_hiddens_1, num_hiddens_2))
    R = normal((num_hiddens_1, num_hiddens_2))
    T = normal((num_hiddens_2, num_hiddens_2))
    s_2 = torch.zeros(num_hiddens_2, device=device)

    # 输出层参数
    Q = normal((num_hiddens_2, num_outputs))
    s_3 = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [U, W, s_1, V, R, T, s_2, Q, s_3]
    for param in params:
        param.requires_grad_(True)
    return params


# 一开始没有隐变量，使用全 0 置位
def init_wrnn_state(batch_size, num_hiddens_1, num_hiddens_2, device):
    return (torch.zeros((batch_size, num_hiddens_1), device=device),
            torch.zeros((batch_size, num_hiddens_1), device=device))


def w_rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    U, W, s_1, V, R, T, s_2, Q, s_3 = params
    A, B = state
    outputs = []
    # X的形状：(批量大小，词表大小)，和上面 inputs 的形状对应
    for X in inputs:
        A_old = A.clone()
        A = torch.tanh(torch.mm(X, U) + torch.mm(A_old, W) + s_1)
        B = torch.tanh(torch.mm(A, V) + torch.mm(A_old, R) + torch.mm(B, T) + s_2)
        O = torch.relu(torch.mm(B, Q) + s_3)
        outputs.append(O)
    # outputs 是个列表 (nums_step, batch_size, len(vocab))，需要上下拼接成二维张量 (nums_step x batch_size, len(vocab))
    # 然后在和 (A,B) 左右拼接成元组，这里拼接是为了方便封装
    return torch.cat(outputs, dim=0), (A, B)


class WRNNModelScratch(RNNModelScratch):
    def __init__(self, vocab_size, num_hiddens_1, num_hiddens_2, device):
        self.vocab_size, self.num_hiddens_1, self.num_hiddens_2 = vocab_size, num_hiddens_1, num_hiddens_2
        self.params = get_params(vocab_size, num_hiddens_1, num_hiddens_2, device)

    def __call__(self, X, state):
        # 颠倒 X，使之能够获得（时间步数，批量大小，词表大小）的 X，最后个维度就是独热编码
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return w_rnn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return init_wrnn_state(batch_size, self.num_hiddens_1, self.num_hiddens_2, device)
