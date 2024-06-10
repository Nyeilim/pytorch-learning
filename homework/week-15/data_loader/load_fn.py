from data_loader.dataLoader import DataLoader
from data_loader.vocabulary import Vocabulary


# 返回小说的词元索引列表和词表
def load_corpus(max_tokens=-1):
    lines = read_novel()
    tokens = tokenize(lines, 'char')
    vocab = Vocabulary(tokens)
    # 将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# 加载数据
def load_data(batch_size, num_steps, max_tokens=10000):
    data_iter = DataLoader(batch_size, num_steps, max_tokens, load_corpus)
    return data_iter, data_iter.vocab


# 拆分词元
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


# 读取小说
def read_novel():  # @save

    with open('./novel/shediaoyingxiongchuan_jinyong.txt', 'r') as f:
        novel_1 = f.readlines()
    with open('./novel/shendiaoxialv_jinyong.txt', 'r') as f:
        novel_2 = f.readlines()
    with open('./novel/tianlongbabu_jinyong.txt', 'r') as f:
        novel_3 = f.readlines()
    with open('./novel/xiaoaojianghu_jinyong.txt', 'r') as f:
        novel_4 = f.readlines()
    with open('./novel/xueshanfeihu_jinyong.txt', 'r') as f:
        novel_5 = f.readlines()
    with open('./novel/yitiantulongji_jinyong.txt', 'r') as f:
        novel_6 = f.readlines()

    lines = novel_1 + novel_2 + novel_3 + novel_4 + novel_5 + novel_6
    # 去掉空行，去掉换行符
    return [line.strip() for line in lines if len(line.strip()) != 0]
