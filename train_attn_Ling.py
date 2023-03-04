import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import csv
import sys
import json
import math
from sentence_transformers import SentenceTransformer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertTokenizer
from nltk.stem import WordNetLemmatizer
from rnnattn_wd import RNNAttnCls2
from torch.autograd import Variable
import copy


bert_model = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "bert-base-uncased.tar.gz")

model = BertForMaskedLM.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.cuda()
model.eval()
bert_embeddings = model.bert.embeddings.word_embeddings
bert_embeddings.weight.requires_grad = False
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case="True")

cudadevice = torch.device("cuda")
cls_save_path = r"./SentiCls"
batch_size = 8
train_learning_rate = 1e-3
train_epoch = 10


class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.emb_dim = copy.deepcopy(bert_embeddings.weight.shape[-1])
        self.Linear = nn.Linear(768, 3)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = x.to(dtype = torch.long)
        x = bert_embeddings(x)
        batchsize = x.size(0)
        x = x.view(-1, x.size(-1))
        x = self.Linear(x)
        x = self.softmax(x)
        x = x.view(batchsize, -1, x.size(-1))
        return x




def get_sentiment_words():
    with open(file=global_config.sentiment_words_file_path,
              mode='r', encoding='ISO-8859-1') as sentiment_words_file:
        words = sentiment_words_file.readlines()
    words = set(word.strip() for word in words)

    return words

params = {
        "EPOCH": 100,
        "CLASS_SIZE": 2,
        "H_DIM":32
    }

params2 = {
        "EPOCH": 100,
        "CLASS_SIZE":2,
        "H_DIM":32
    }

def load_dataset():
    data = open(r"./raw_data/amazon/sentiment.dev.0",'r')
    lines = []
    for l in data:
	    lines.append(l)
    data.close()  
    data = open(r"./raw_data/amazon/sentiment.dev.1",'r')
    for l in data:
	    lines.append(l)
    data.close()
    rec = [[], []]
    recsensi = []
    recunsensi = []

    with open(file=r'./src_Ling/opinion-lexicon-English/positive-words.txt',
              mode='r', encoding='ISO-8859-1') as sentiment_words_file:
        positive = sentiment_words_file.readlines()
        positive = [line.rstrip() for line in positive]
    

    with open(file=r'./src_Ling/opinion-lexicon-English/negative-words.txt',
              mode='r', encoding='ISO-8859-1') as sentiment_words_file:
        negative = sentiment_words_file.readlines()
        negative = [line.rstrip() for line in negative]
    
    maxlength = -1

    for l in lines:
        tokens = tokenizer.tokenize(l)
        rec[0].append(' '.join('%s' %id for id in tokens))
        temp = []
        tempsensi = []
        tempunsensi = []
        tempsensi2 = []
        if maxlength < len(tokens):
            maxlength = len(tokens)
        for token in tokens:
            if token in positive:
                tempsensi.append(token)
                temp.append(1)
            elif token in negative:
                tempsensi2.append(token)
                temp.append(1)
            else:
                tempunsensi.append(token)
                temp.append(0)
        rec[1].append(temp)
        
        recsensi.append(' '.join('%s' %id for id in tempsensi))
        recsensi.append(' '.join('%s' %id for id in tempsensi2))
        recunsensi.append(' '.join('%s' %id for id in tempunsensi))
        # print(rec)

    return rec, maxlength + 10


def cal_context_loss(choose, unchoose, pred_choose, pred_unchoose, attn, device):
    loss_choose = torch.zeros(1).to(device)
    loss_unchoose = torch.zeros(1).to(device)
    # print(pred_choose)
    # print(pred_unchoose)
   
    if len(choose) > 0:
        judge_choose, _ = attn(choose.unsqueeze(0))
        judge_choose = judge_choose[0]
        loss_choose = torch.sum(torch.mul(-judge_choose[0], torch.log(pred_choose[:, 0]))) + torch.sum(torch.mul(-judge_choose[1], torch.log(pred_choose[:, 1])))
        # print("asfda")
        # print(torch.sum(torch.mul(-judge_choose[0], torch.log(pred_choose[:, 0]))))
        # print(torch.sum(torch.mul(-judge_choose[1], torch.log(pred_choose[:, 1]))))
    
    if len(unchoose) > 0:
        judge_unchoose, _ = attn(unchoose.unsqueeze(0))
        judge_unchoose = judge_unchoose[0]
        
        loss_unchoose = torch.sum(torch.mul(-judge_unchoose[0], torch.log(pred_unchoose[:, 0]))) + torch.sum(torch.mul(-judge_unchoose[1], torch.log(pred_unchoose[:, 1])))
        # print("adsf")
        # print(torch.sum(torch.mul(-judge_unchoose[0], torch.log(pred_unchoose[:, 0]))))
        # print(torch.sum(torch.mul(-judge_unchoose[1], torch.log(pred_unchoose[:, 1]))))


    return loss_choose + loss_unchoose


def cal_senti_loss(prediction, data, clss):
    data = data.view(-1)
    prediction = prediction.view(-1, 2)
    sentiout = clss(data.unsqueeze(0))[0]
    # print(data.unsqueeze(0))
    # print(sentiout)
    
    # print(sentiout.size())

    return -torch.sum(torch.mul((sentiout[:, 0] + sentiout[:, 2]), torch.log(prediction[:, 1]))) - torch.sum(torch.mul(sentiout[:, 1], torch.log(prediction[:, 0])))

def train_cls(dataset, model, attn, clss, device):
    model.train()
    train_set = dataset
    # testset, trainset = torch.utils.data.random_split(dataset, [math.floor(0.2 * len(dataset)), len(dataset) - math.floor(0.2 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    total_loss, correct = 0.0, 0.0
    criterion = nn.CrossEntropyLoss(ignore_index = -1)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_learning_rate)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data)
        # print(target)
        data, target = data.to(device), target.to(device)
        model = model.to(device)
        attn = attn.to(device)
        clss = clss.to(device)
        optimizer.zero_grad()
        _, prediction = model(data)
        # print(prediction)
        pred = prediction.argmax(dim=2, keepdim=True)
        loss = torch.zeros(1).to(device)
        for i in range(batch_size):
            with torch.no_grad():
                index = []
                unindex = []
                length = -1
                for j in range(len(data[0])):
                    if target[i][j] == -1:
                        length = j
                        break
                    if pred[i][j].item() == 1:
                        index.append(j)
                    else:
                        unindex.append(j)
            # print(index)
            # print(unindex)
            choose = data[i][index]
            pred_choose = prediction[i][index]
            unchoose = data[i][unindex]
            pred_unchoose = prediction[i][unindex]
            contextloss = cal_context_loss(choose, unchoose, pred_choose, pred_unchoose, attn, device)
            # print("daf")
            # print(contextloss)
            sentiloss = cal_senti_loss(prediction[i], data[i], clss)
            # print(sentiloss)
            loss += 0.2 * contextloss + 0.9 * sentiloss   #可以调
            # print(loss)


        
        loss.backward()        


        optimizer.step()
            
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print("Index: " + str(batch_idx * batch_size) + " / " + str(len(dataset)) + " Loss: " + str(loss))
        # pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        # correct += pred.eq(target.view_as(pred)).sum().item()
    
    total_loss /= len(train_loader.dataset)

    return total_loss





def main():
    rec, maxlength = load_dataset()

    # print(rec[0])
    clsdata = torch.Tensor(tokenizer.batch_encode_plus(rec[0], max_length = maxlength, padding='max_length', truncation=True)['input_ids'])
    for i in range(len(rec[1])):
        rec[1][i].extend([-1] * (maxlength - len(rec[1][i])))
    clstarget = torch.Tensor(rec[1]).to(torch.int64)
    # for i in clsdata:
    #     print(len(i))
 
    
    cls_dataset = torch.utils.data.TensorDataset(clsdata, clstarget)




    attnlstm = RNNAttnCls2(**params2)

    trained_attn = RNNAttnCls2(**params)
    checkpoint = torch.load(cls_save_path + '/attclssensi.pth')
    trained_attn.load_state_dict(checkpoint)
    trained_cls = LinearClassifier()
    checkpoint = torch.load(cls_save_path + '/clsenti.pth')
    trained_cls.load_state_dict(checkpoint)
    
    print("Start LSTM Training")
    for i in range(train_epoch):
        loss = train_cls(cls_dataset, attnlstm, trained_attn, trained_cls, cudadevice)
        print("-----------------------------------------")
        print("Epoch: " + str(i + 1) + "    Avg. Loss:  " + str(loss))
    print("End LSTM Training")


    print("Saving Models...")
    torch.save(attnlstm.state_dict(), cls_save_path + '/attnlstm.pth')

    print("Complete!")



main()