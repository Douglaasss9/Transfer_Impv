# Transfer Impv

## preprocess impv

* LARE method
* prepare file:
  * download gloss_embedding.npy in https://cloud.tsinghua.edu.cn/d/f6baaff5c398463388b2/ to src_LARE/
  * download SentiWordNet_3.0.0.txt in https://github.com/aesuli/SentiWordNet to src_LARE/
  * unzip stanford-postagger-full-2020-11-17 in https://nlp.stanford.edu/software/tagger.html to src_LARE/stanford-postagger-full-2020-11-17
* tune the bound rate in preprocess_LARE.py
  * As Boundrate grows, the tokens in the sent are more likely to be neither negative and positive
* run 

```
bash ./scripts/LARE/preprocess_amazon_LARE.sh
bash ./scripts/LARE/preprocess_yelp_LARE.sh
```

to preprocess the raw data using LARE method

In fact, the preprocessed data are contained in the project

***

### run finetune

run

```
bash ./scripts/LARE/fine_tune_amazon_LARE.sh
```

or

```
bash ./scripts/LARE/fine_tune_yelp_LARE.sh
```

to run the finetune process.

## Sentiment Enhanced Finetune Method

* This Method aims to improve the performance of transferring BERT by adding sequence-level sentiment instructions for BERT training

### Methodology

#### Classifier Model

* We choose a Attention-based LSTM model as our sentiment classifier.
* The Attention-based LSTM takes the whole sentence as the input and outputs a score of sentiment judgement.

#### Methodology

1. During the fine-tune-bert process, we train the classifier with BERT taking the output of BERT as the embedding.
2. The target of the training process is the sentiment label which is written before by different methods.
3. The training optimizer is Adam with standard setting and the criterion is Cross Entropy Loss.
4. We save the model after the fine-tune-bert process, and in the fine-tine-cbert process, we load the model.
5. During the fine-tune-cbert process, the BERT model will not only instructed by the labels themselves and also the judgements from the classifier.
6. The added loss is calculated by the output of classifier and the anti-label with Cross Entropy Loss.

#### Using Instruction

1. The added loss weight is set as 1e-5, which is "alpha" at the top of cbert file.
2. The saving directory is ./SentiCls
3. Just replace the fine_tune_bert.py and fine_tune_cbert.py by fine_tune_bert_senticls.py and fine_tune_cbert_senticls.py in the commands

## LING Label Sentiment Method

* It is a total new method to label the raw data considers both the sentiment information and context information of the token sets.

#### Preparation

* we use the "opinion-lexicon-English" word list as sentiment word instruction which is contains in "src_Ling"

#### Pipeline

![3950534006b568cb4ab17496503d153](https://github.com/Douglaasss9/Transfer_Impv/raw/main/Transfer%20Impv.assets/3950534006b568cb4ab17496503d153.png)

#### Classifier Model

1. A MLP with independent Bert embedding part and 3-dimension output.
2. Two Attention-based LSTM with independent Bert embedding part and 2-demension output

#### Methodology

1. Train the preprocessed classifier model
   1. label the raw data with lexicon sentiment word list
   2. train the MLP (token sentiment cls)
      1. tokenize the sentences into words
      2. the label is prepared
      3. balance the label: repeat the sentiment words till their number is equal to non-sentiment words.
      4. train the MLP
      5. the optimizer is Adam
   3. train the Attn-based LSTM 1
      1. divide the tokens of each sentence into two sets: sentiment words and non-sentiment words
      2. label the sentimens words list as 1, and label the others as 0
      3. train the Attn-based LSTM taking the sequential sets as input and output a score
      4. the optimizer is Adam
2. Train the classifier model
   1. the input of the LSTM 2 is raw sentences
   2. the loss combines two parts:
      1. sentiment loss is calculated by cross-entropy loss between the output of classifier model and the output of the pretrained MLP
      2. context loss is calculated by cross-entropy loss between the output of the pretrained LSTM 1 and the output of class model for both the selected set and unselected set of LSTM 2
   3. the optimizer is Adam

3. preprocess the raw_data using pretrained LSTM

#### Instruction

1. The learning rate of each training process and the training epoch can be set in the top of train_Ling.py and train_attn_Ling.py
2. The training file can be set in train_Ling.py and train_attn_Ling.py
3. The restore path is ./SentiCls
4. The run step:
   1. python train_Ling.py
   2. python train_attn_Ling.py
   3. bash ./scripts/Ling/preprocess_amazon_Ling.sh
      bash ./scripts/Ling/preprocess_yelp_Ling.sh
   4. bash ./scripts/Ling/fine_tune_amazon_Ling.sh or bash ./scripts/Ling/fine_tune_yelp_Ling.sh

