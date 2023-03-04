import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
cls_learning_rate = 1e-3
cls_epoch = 10
senti_learning_rate = 1e-3
senti_epoch = 2

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
        tokens = l.split()
        # print(tokens)
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
                temp.append(2)
            elif token in negative:
                tempsensi2.append(token)
                temp.append(0)
            else:
                tempunsensi.append(token)
                temp.append(1)
        rec[1].append(temp)
        
        recsensi.append(' '.join('%s' %id for id in tempsensi))
        recsensi.append(' '.join('%s' %id for id in tempsensi2))
        recunsensi.append(' '.join('%s' %id for id in tempunsensi))
        # print(rec)

    return rec, maxlength + 10, recsensi, recunsensi

def train_cls(dataset, model, device):
    model.train()
    train_set = dataset
    # testset, trainset = torch.utils.data.random_split(dataset, [math.floor(0.2 * len(dataset)), len(dataset) - math.floor(0.2 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    total_loss, correct = 0.0, 0.0
    criterion = nn.CrossEntropyLoss(ignore_index = -1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cls_learning_rate)
    recnum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data)
        
        data, target = data.to(device), target.to(device)
        lengthall = 0
        indexsensi = []
        target = target.view(-1)
        data = data.view(-1)
        for i in range(len(target)):
            if target[i] != -1:
                lengthall += 1
            if target[i] == 0 or target[i] == 2:
                indexsensi.append(i)

        targetsensi = target[indexsensi]
        datasensi = data[indexsensi]
        repeatnum = math.floor((lengthall - len(indexsensi)) / len(indexsensi))
        data = torch.concat([data[:lengthall], datasensi.repeat(repeatnum)], dim=0)
        target = torch.concat([target[:lengthall], targetsensi.repeat(repeatnum)], dim=0)

        
        model = model.to(device)
        optimizer.zero_grad()
        prediction = model(data).squeeze(1)
        


        loss = criterion(prediction.view(-1, 3), target.view(-1))
        loss.backward()        
        pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        recnum += len(target)
        optimizer.step()
            
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print("Index: " + str(batch_idx * batch_size) + " / " + str(len(dataset)) + " Loss: " + str(loss) + "   Acc:    " + str(correct / recnum))

    
    total_loss /= len(train_loader.dataset)
    

    return total_loss, correct / recnum



def train_attn(dataset, model, device):
    model.train()
    train_set = dataset
    # testset, trainset = torch.utils.data.random_split(dataset, [math.floor(0.2 * len(dataset)), len(dataset) - math.floor(0.2 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    total_loss, correct = 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cls_learning_rate)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model = model.to(device)
        optimizer.zero_grad()
        prediction, _ = model(data)
        loss = criterion(prediction, target)
        loss.backward()        


        optimizer.step()
            
        total_loss += loss.item()
        pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 100 == 0:
            print("Index: " + str(batch_idx * batch_size) + " / " + str(len(dataset)) + " Loss: " + str(loss) + "  Avg. acc: " + str(correct / ((batch_idx + 1) * batch_size)))

    
    total_loss /= len(train_loader.dataset)
    correct /= len(train_loader.dataset)

    return total_loss, correct

def main():
    rec, maxlength, recsensi, recunsensi = load_dataset()
    # print(rec[0])
    clsdata = torch.Tensor(tokenizer.batch_encode_plus(rec[0], max_length = maxlength, padding='max_length', truncation=True)['input_ids'])
    for i in range(len(rec[1])):
        rec[1][i].extend([-1] * (maxlength - len(rec[1][i])))
    clstarget = torch.Tensor(rec[1]).to(torch.int64)
    # for i in clsdata:
    #     print(len(i))
    sensidata = torch.Tensor(tokenizer.batch_encode_plus(recsensi, max_length = maxlength, padding='max_length', truncation=True)['input_ids'])
    unsensidata = torch.Tensor(tokenizer.batch_encode_plus(recunsensi, max_length = maxlength, padding='max_length', truncation=True)['input_ids'])
    sensitarget = [1] * len(sensidata)
    sensitarget.extend([0] * len(unsensidata))
    # print(len(unsensidata))
    sensitarget = torch.Tensor(sensitarget).to(torch.int64)
    
    cls_dataset = torch.utils.data.TensorDataset(clsdata, clstarget)
    sensidataset = torch.utils.data.TensorDataset(torch.cat([sensidata, unsensidata], dim=0), sensitarget)



    attclssensi = RNNAttnCls2(**params)
    clsenti = LinearClassifier()
    print("Start CLS Training")
    for i in range(cls_epoch):
        loss, acc = train_cls(cls_dataset, clsenti, cudadevice)
        print("-----------------------------------------")
        print("Epoch: " + str(i + 1) + "    Avg. Loss:  " + str(loss) + "   Avg. acc: " + str(acc))
    print("End CLS Training")
    print("Start Attn Training")
    for i in range(senti_epoch):
        loss, acc = train_attn(sensidataset, attclssensi, cudadevice)
        print("-----------------------------------------")
        print("Epoch: " + str(i + 1) + "    Avg. Loss:  " + str(loss) + "   Avg. acc: " + str(acc))
    print("End Attn Training")

    print("Saving Models...")
    torch.save(clsenti.state_dict(), cls_save_path + '/clsenti.pth')
    torch.save(attclssensi.state_dict(), cls_save_path + '/attclssensi.pth')
    print("Complete!")



main()