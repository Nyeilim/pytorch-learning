from rnn import *
from w_rnn import *
from train_fn import *

num_hiddens = 512  # 隐藏层
num_epochs, lr = 50000, 1  # 迭代次数，学习率
batch_size, num_steps = 3, 4  # 批量大小，时间步
train_iter, vocab = load_data(batch_size, num_steps)  # 每次采样的小批量数据形状是二维张量：（批量大小，时间步数）。加载词表和迭代器

rnn = RNNModelScratch(len(vocab), num_hiddens, 'cuda')
train(rnn, train_iter, vocab, lr, num_epochs, 'cuda', use_random_iter=True)

num_hiddens_1, num_hiddens_2 = 512, 512
w_rnn = WRNNModelScratch(len(vocab), num_hiddens_1, num_hiddens_2, 'cuda')
train(rnn, train_iter, vocab, lr, num_epochs, 'cuda', use_random_iter=True)
