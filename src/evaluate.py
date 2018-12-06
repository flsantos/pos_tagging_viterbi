import pickle
from sklearn.model_selection import KFold
import numpy as np
import logging
import datetime

from viterbi import *


# Read phrases (pre-processed)
with open('preprocessed_CETEN_v2.pkl', 'rb') as input:
    phrases = pickle.load(input)


# Define logging file
logging.basicConfig(filename='evaluation.log', level=logging.INFO)
def log(msg):
    logging.info(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")+ ': '+str(msg))


#Define KFold split
kf = KFold(n_splits=5, random_state=42, shuffle=True)


# Evaluate Viterbi
phrases_data = np.array(phrases)
round = 0
for train_index, test_index in kf.split(phrases_data):
    round=round+1
    print('CV Round '+str(round))
    log('CV Round '+str(round))
    
    print('Train test split: '+ str(len(train_index))+','+str(len(test_index)))
    log('Train test split: '+ str(len(train_index))+','+str(len(test_index)))
    phrases_train = phrases_data[train_index]
    phrases_test = phrases_data[test_index]
    
    
    print('Counting word frequency...')
    log('Counting word frequency...')
    word_freq, word_counts, tags_counter, allowed_tags = get_train_counts(phrases_train)
    
    
    print('Calculating emission probabilities...')
    log('Calculating emission probabilities...')
    e_prob = get_emission_probs(word_freq, tags_counter)
    print('Calculating transition probabilities...')
    log('Calculating transition probabilities...')
    q_prob = get_transition_probs(phrases_train,word_freq,tags_counter)
    
    print('Viterbi...')
    log('Viterbi...')
    corretas_viterbi, corretas_baseline, totais, elapsed = evaluate_viterbi(word_freq,\
                                                                            word_counts,\
                                                                            allowed_tags,\
                                                                            e_prob, q_prob,\
                                                                            phrases_test,\
                                                                            sample=-1,\
                                                                            log_fn=log)
    
    print(corretas_viterbi, corretas_baseline, totais, elapsed)
    log(str([corretas_viterbi, corretas_baseline, totais, elapsed]))
    
    print("")
    log("")
    
logging.shutdown()