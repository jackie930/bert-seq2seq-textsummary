## THUCNews 原始数据集
import sys
sys.path.append("../")
sys.path.append("../bert_seq2seq/")

import torch 
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
import numpy as np
import os
import json
import time
import glob
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader

from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert

vocab_path = "./state_dict/bert-base-chinese-vocab.txt"  # 模型字典的位置
word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)
model_name = "bert"  # 选择模型名字
model_path = "./state_dict/pytorch_model.bin"  # 模型位置
recent_model_path = "./state_dict/bert_auto_title_model-sports.bin"   # 用于把已经训练好的模型继续训练
model_save_path = "./state_dict/bert_auto_title_model-sports.bin"
batch_size = 16
lr = 1e-5
maxlen = 256

class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self,txt_folder) :
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        ## 拿到所有文件名字
        self.txts = glob.glob(txt_folder+'/*.txt')
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        text_name = self.txts[i]
        with open(text_name, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.split('SPLIT')
        if len(text) > 1:
            title = text[0]
            content = text[1]
            token_ids, token_type_ids = self.tokenizer.encode(
                content, title, max_length=maxlen
            )
            output = {
                "token_ids": token_ids,
                "token_type_ids": token_type_ids,
            }
            return output

        self.__getitem__(i + 1)

    def __len__(self):

        return len(self.txts)
        
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

class Trainer:
    def __init__(self,txt_folder):
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name)
        ## 加载预训练的模型参数～
        
        self.bert_model.load_pretrain_params(model_path, keep_tokens=keep_tokens)
        # 加载已经训练好的模型，继续训练

        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(txt_folder)
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 100 == 0:
                self.bert_model.eval()
                test_data = ["新浪财经讯 3月9日下午消息 花旗银行(中国)有限公司(花旗中国)今日宣布，联手亨德森全球投资(亨德森)，推出花旗银行代客境外理财产品——亨德森全球远见基金系列。该产品系列的推出标志着花旗银行成为中国首家提供代客境外理财产品——房地产海外基金的银行。",
                             "市场信心恢复的一个主要标志，就是投资者由过去对金融危机的过度恐慌，转变到当前对金融危机的发展过程以及金融危机对实体经济的危害程度有了比较清醒的认识。如随着美国次贷危机转变为全球性金融危机，道琼斯股票指数跌势明显趋缓。"]
                for text in test_data:
                    print(self.bert_model.generate(text, beam_size=3))
                print("loss is " + str(report_loss))
                report_loss = 0
                # self.eval(epoch)
                self.bert_model.train()
            if step % 8000 == 0:
                self.save(model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                               
                                                )
            report_loss += loss.item()
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        self.save(model_save_path)

if __name__ == '__main__':
    txt_folder = '../data/sports'
    trainer = Trainer(txt_folder)
    train_epoches = 20

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)