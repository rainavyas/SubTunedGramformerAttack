'''
At input requires aligned .corr, .pred and .inc files

Analyses the impact of the attack:

1) report average edits between .inc and .pred
2) report average edits between .inc and .pred, filtering for samples to contain at least one of the target words
'''

import sys
import os
import argparse
from gec_tools import count_edits
from statistics import mean, stdev

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('PRED', type=str, help='Path to predicted output test data')
    commandLineParser.add_argument('CORR', type=str, help='Path to correct output test data')
    commandLineParser.add_argument('--target', default='', type=str, help='target words')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analyse_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # get target words as a list
    target_words = args.target
    if target_words == '':
        target_words = []
    else:
        target_words = target_words.split()
        target_words = [" "+t for t in target_words]
    
    # Load files
    with open(args.INC, 'r') as f:
        lines = f.readlines()
    incs = [l.rstrip('\n') for l in lines]
    with open(args.PRED, 'r') as f:
        lines = f.readlines()
    preds = [l.rstrip('\n') for l in lines]
    with open(args.CORR, 'r') as f:
        lines = f.readlines()
    corrs = [l.rstrip('\n') for l in lines]
    
    # Get average counts for all samples and filters
    edit_counts_all = []
    edit_counts_filtered = []
    for i, (inc, pred) in enumerate(zip(incs, preds)):
        num_edits = count_edits(inc, pred)
        edit_counts_all.append(num_edits)
        if any([t in inc for t in target_words]):
            edit_counts_filtered.append(num_edits)

    # Report stats
    print('-----------')
    print(f'Average number of edits: {mean(edit_counts_all)} +- {stdev(edit_counts_all)}')
    print('-----------')
    print(f'Number of samples filtered for target words {target_words}: {len(edit_counts_filtered)}')
    print(f'Average number of edits filtered {mean(edit_counts_filtered)} +- {stdev(edit_counts_filtered)}')
    print('-----------')
    
    