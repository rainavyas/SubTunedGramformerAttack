'''
    Check for sentiment bias in GEC data
    Generate retention curve:
        - order input sentences from least to most positive as per some sentiment classifier
        - create two curves: for every fraction of retention
            - cumulative edits between input and prediction
            - cumulative edits between input and correction

    Need three files at the inputs:
        .inc, .pred, .corr, where each file is of the following structure:

            sent1
            sent2
              .
              .
              .
    This structure is created by the output files of eval_uni_attack.py
'''

from cProfile import label
import sys
import os
import argparse
from gec_tools import count_edits
from sentiment_models import BertSequenceClassifier
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def load_sentences(filepath):
    with open(filepath, 'r') as f:
        sens = f.readlines()
    sens = [s.rstrip('\n') for s in sens]
    return sens


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('INC', type=str, help='Path to input data')
    commandLineParser.add_argument('CORR', type=str, help='Path to correct output test data')
    commandLineParser.add_argument('PRED', type=str, help='Path to predictions for test data')
    commandLineParser.add_argument('MODEL', type=str, help='Sentiment Model')
    commandLineParser.add_argument('OUT', type=str, help='output file to save figure')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/sentiment_bias.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the sentiment model
    model = BertSequenceClassifier()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load sentences
    inc_sens = load_sentences(args.INC)
    corr_sens = load_sentences(args.CORR)
    pred_sens = load_sentences(args.PRED)

    # get edits
    pred_edits = []
    corr_edits = []
    for count, (i,c,p) in enumerate(zip(inc_sens, corr_sens, pred_sens)):
        print(f'Counting edits {count}/{len(inc_sens)}')
        pred_edits.append(count_edits(i,p))
        corr_edits.append(count_edits(i,c))
    
    # calculate positive sentiment per inc sentence
    sentiments = []
    sf = nn.Softmax(dim=0)
    for count, sen in enumerate(inc_sens):
        print(f'Getting sentiment {count}/{len(inc_sens)}')
        encoded = model.tokenizer([sen], return_tensors='pt')
        ids = encoded['input_ids']
        logits = model(ids).squeeze()
        probs = sf(logits).tolist()
        sentiments.append(probs[1])

    # order by sentiment
    items = [(s,c,p) for s,c,p in zip(sentiments, corr_edits, pred_edits)]
    ordered_items = sorted(items, key=lambda x: x[0])
    ord_pred_edits = [o[2] for o in ordered_items]
    ord_corr_edits = [o[1] for o in ordered_items]
    
    # retention plots
    fracs = [(i+1)/len(items) for i,_ in enumerate(items)]
    pred_cum_edits = [0]
    corr_cum_edits = [0]
    for i in range(len(ordered_items)):
        pred_cum_edits.append(ord_pred_edits[i]+pred_cum_edits[i])
        corr_cum_edits.append(ord_corr_edits[i]+corr_cum_edits[i])
    pred_cum_edits = [e/pred_cum_edits[-1] for e in pred_cum_edits[1:]]
    corr_cum_edits = [e/corr_cum_edits[-1] for e in corr_cum_edits[1:]]



    plt.plot(fracs, pred_cum_edits, label='pred')
    plt.plot(fracs, corr_cum_edits, label='corr')
    plt.ylabel('Cumulative Edits (%)')
    plt.xlabel('Retention Fraction')
    plt.legend()
    plt.savefig(args.OUT, bbox_inches='tight')






