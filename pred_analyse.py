'''
Predict after sub attack and analysis in one script
'''

import sys
import os
import argparse
import torch
from gec_tools import get_sentences, correct, count_edits
from Seq2seq import Seq2seq
import json
from datetime import date
from uni_attack import substitute
from statistics import mean, stdev

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('SUB_DICT', type=str, help='path to json dict with substitutions')
    commandLineParser.add_argument('OUT', type=str, help='Path to output file')
    commandLineParser.add_argument('--target', default='', type=str, help='target replaced words')
    commandLineParser.add_argument('--seed', type=int, default=1, help='reproducibility')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pred_analyse.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)

    # Load Model
    model = Seq2seq()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load input sentences
    _, sentences = get_sentences(args.IN)

    # Load sub dict
    with open(args.SUB_DICT) as json_file:
        sub_dict = json.load(json_file)

    # get target words as a list
    target_words = args.target
    if target_words == '':
        target_words = [' ']
    else:
        target_words = target_words.split()

    # Get average counts for all samples and filters
    edit_counts_all = []
    edit_counts_filtered = []
    
    for i, sent in enumerate(sentences):
        set_seeds(args.seed)
        print(f'On {i}/{len(sentences)}')
        sent = substitute(sent, sub_dict)
        pred = correct(model, sent)
        num_edits = count_edits(sent, pred)
        edit_counts_all.append(num_edits)
        if any([(((" "+t+" " in sent) or (" "+t+"." in sent)) or (" "+t+"," in sent)) for t in target_words]):
            edit_counts_filtered.append(num_edits)

    # Report stats
    text = ''
    text += '-----------'
    text += f'\nAverage number of edits: {mean(edit_counts_all)} +- {stdev(edit_counts_all)}'
    text += '\n-----------'
    text += f'\nNumber of samples filtered for target words {target_words}: {len(edit_counts_filtered)}'
    try:
        text += f'\nAverage number of edits filtered {mean(edit_counts_filtered)} +- {stdev(edit_counts_filtered)}'
    except:
        text += '\n-----------'
    print(text)

    with open(args.OUT, 'w') as f:
        f.write(text)