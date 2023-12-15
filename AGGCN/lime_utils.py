import random
import argparse
import tqdm
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import numpy as np
import pandas as pd

import stanza
import json 

from pre_processing_stanza import pre_processing

class_names= ['Other', 'Entity-Destination', 'Cause-Effect', 'Member-Collection', 'Entity-Origin', 'Message-Topic', 'Component-Whole', 'Instrument-Agency', 'Product-Producer', 'Content-Container']

example_relation = "Message-Topic"

#stanza_client = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse ', be_quiet=True)



def evaluation_lime(dataset_): 
    model_dir = "saved_models/01"
    model = 'checkpoint_epoch_150.pt'
    data_dir = 'dataset/semeval'
    dataset = dataset_#'sample'
    seed = 1234
    cuda = torch.cuda.is_available()
    cpu = 'store_true'

    torch.manual_seed(seed)
    random.seed(1234)
    if cpu:
        cuda = False
    elif cuda:
        torch.cuda.manual_seed(seed)

    # load opt
    model_file = model_dir + '/' + model
    #print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)
    trainer = GCNTrainer(opt)
    trainer.load(model_file)

    # load vocab
    vocab_file = model_dir + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

    # load data
    data_file = opt['data_dir']  + '/{}.json'.format(dataset)
    #print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
    batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

    #helper.print_config(opt)
    label2id = constant.LABEL_TO_ID
    id2label = dict([(v,k) for k,v in label2id.items()])

    predictions = []
    all_probs = []
    batch_iter = tqdm(batch)
    
    for i, b in enumerate(batch_iter):
        preds, probs, _ = trainer.predict(b)
        predictions += preds
        all_probs += probs

    predictions = [id2label[p] for p in predictions]
    #print(predictions)
    #p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
    #print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(dataset,p,r,f1))

    #print("Evaluation ended.")

    return all_probs, predictions


""" 
def lime_pipeline(texts):
    print("obj_start: ", obj_start)
    number_of_texts = len(texts)

    #creates json file in 'dataset/semeval/pre_processed_data.json'
    pre_processed_sample = pre_processing(texts=texts,relation= "Cause-Effect")

    #uses the new created file
    #probability, label = evaluation_lime(dataset_='pre_processed_data_')
    probability, label = evaluation_lime(dataset_='pre_processed_data_')

    reshaped_probs = np.array(probability).reshape(number_of_texts, 10)

    return reshaped_probs """

def process_json(file_path):
    df = pd.read_json(file_path)
    df = df[['id', 'relation', 'token', 'subj_start', 'obj_start']]

    df['sentence'] = [' '.join(text) for text in  df['token']]

    return df


