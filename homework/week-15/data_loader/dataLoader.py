import random
import torch


# 随机采样方法
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]  # 确定初始偏移后切片
    num_subseqs = (len(corpus) - 1) // num_steps  # 能切多个片
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))  # 每个切片的起始索引
    random.shuffle(initial_indices)  # 打乱切片

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]  # 返回切片内容

    num_batches = num_subseqs // batch_size  # 这么多片能分多少批
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]  # 切片的内容作为 X
        Y = [data(j + 1) for j in initial_indices_per_batch]  # 切片的内容往后挪一步作为预测标签 Y
        yield torch.tensor(X), torch.tensor(Y)


class DataLoader:
    """加载序列数据的迭代器"""

    def __init__(self, batch_size, num_steps, max_tokens, load_corpus):
        self.corpus, self.vocab = load_corpus(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return seq_data_iter_random(self.corpus, self.batch_size, self.num_steps)
