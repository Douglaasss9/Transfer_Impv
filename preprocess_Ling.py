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

from torch.autograd import Variable
import os
import torch
import numpy as np
import torch.nn.functional as F
from utils import read_data, clean_str, load_cls, load_vocab
from functools import cmp_to_key
import sys
import json

bert_model = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "bert-base-uncased.tar.gz")

model = BertForMaskedLM.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.cuda()
model.eval()
bert_embeddings = model.bert.embeddings.word_embeddings

bert_embeddings.weight.requires_grad = False
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case="True")

cudadevice = torch.device("cuda")
cls_save_path = r"./SentiCls"

params = {
        "EPOCH": 100,
        "CLASS_SIZE": 2,
        "H_DIM":32
    }

attclssensi = RNNAttnCls2(**params)
checkpoint = torch.load(cls_save_path + '/attnlstm.pth')
attclssensi.load_state_dict(checkpoint)
attclssensi.cuda()

with open("run.config", 'rb') as f:
    configs_dict = json.load(f)

task_name = configs_dict.get("task_name")
model_name = configs_dict.get("model_name")
modified = configs_dict.get("modified")
batch_size = 16
max_line = 50000000
operation=sys.argv[2]
def cmp(a, b):
    return (a>b)-(a<b)


for i in attclssensi.parameters():
	i.requires_grad = False
attclssensi.eval()

fr=open(sys.argv[1],'r')
save_path = os.path.join(os.curdir, sys.argv[5], sys.argv[4])
if not os.path.exists(save_path):
	os.mkdir(save_path)
fwname = os.path.join(save_path, sys.argv[3]+'.data.'+operation)
fw=open(fwname,'w')
lines = []
for l in fr:
	lines.append(l)
fr.close()

line_num = min(len(lines), max_line)
for i in range(0, line_num, batch_size):
	batch_range = min(batch_size, line_num - i)
	batch_lines = lines[i:i + batch_range]
	batch_x = [clean_str(sent) for sent in batch_lines]
	maxlength = -1

	for i in batch_x:
		if len(i) > maxlength:
			maxlength = len(i)
	
	maxlength = maxlength + 4
	batch_y = torch.Tensor(tokenizer.batch_encode_plus(batch_lines, max_length = maxlength, padding='max_length', truncation=True)['input_ids'])
	batch_y = batch_y.to(cudadevice)
	pred, attn = attclssensi(batch_y)
	
	pred = np.argmax(attn.cpu().data.numpy(), axis=2)
	# print(pred)

	for line, x, pre, att in zip(batch_lines, batch_x, pred, attn):
		if len(x) > 0:
			# att = att[:len(x)]
			# if task_name == 'yelp':
			# 	avg = torch.mean(att)
			# elif task_name == 'amazon':
			# 	avg = 0.4
			# mask = att.gt(avg)
			# if sum(mask).item() == 0:
			# 	mask = torch.argmax(att).unsqueeze(0)
			# else:
			# 	mask = torch.nonzero(mask.squeeze()).squeeze(1)
			# idx = mask.cpu().numpy()
			
			idx = np.where((pre ==1))[0]
			if len(idx) == 0:
				idx = torch.argmax(att[:, 1]).unsqueeze(0)
				idx = idx.cpu().numpy()

			
			idx = [int(ix) for ix in idx]
			contents = []
			for i in range(0, len(x)):
				if i not in idx:
					contents.append(x[i])
			wl = {"content": ' '.join(contents), "line": line.strip(), "masks": list(idx), "label": str(sys.argv[3][-1])}
			#print(wl)
			wl_str = json.dumps(wl)
			fw.write(wl_str)
			fw.write("\n")

fw.close()
