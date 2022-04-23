'''
Use training data to identify word groups by Part Of Speech (POS) tags
Find the 50 most frequent words in each word group and save files
'''

import argparse
import sys
import os
from gec_tools import get_sentences
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('OUT', type=str, help='Path to dir to save data')
    commandLineParser.add_argument('--num_points', type=int, default=1000, help='Number of training data points to consider')

    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/find_word_groups.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load input sentences
    _, sentences = get_sentences(args.IN, num=args.num_points)

    # dictionary to store POS tag count per word in group
    pos_counter = defaultdict(lambda: defaultdict(int))

    # Populate the POS tag dictionary
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        for (word, tag) in tagged:
            # import pdb; pdb.set_trace()
            pos_counter[tag][word] += 1
    
    # Save each group of words in frequency order
    for pos_tag, word_dict in pos_counter.items():
        word_freqs = []
        for word, freq in word_dict.items():
            word_freqs.append((word, freq))
        word_freqs.sort(key=lambda a:a[1], reverse=True)

        filename = f'{args.OUT}/{pos_tag}.txt'
        with open(filename, 'w') as f:
            f.write("\n".join([f'{word} {freq}' for (word,freq) in word_freqs]))

    