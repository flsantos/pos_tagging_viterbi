from collections import Counter
import nltk
import timeit




#### Definições globais ####
smooth_emission_not_seen = 0
smooth_transition_not_seen = 0

min_threshold = 5 ### frequência mínima para a palavra aparecer na base de teste, caso contrário vira __RARE__
RARE_WORD = '__RARE__'

def get_train_counts(phrases_train):

    ####Converte qualquer palavra com freq < 5 para RARE
    word_counts = Counter()
    for phrase in phrases_train:
        for word in phrase:
            word_counts[word[0]]+=1
            
    word_counts = {word: count for word, count in word_counts.items() if word_counts[word] >= min_threshold}

    phrases_train_rare = []
    for phrase in phrases_train:
        phrases_train_rare.append([(w[0] if word_counts.get(w[0]) else RARE_WORD,w[1]) for w in phrase])


    ####Contagem de palavras na base de treino
    tags_counter = Counter()
    for s in phrases_train_rare:
        for tk in s:
            tag = tk[1]
            if tags_counter[tag]:
                tags_counter[tag]=tags_counter[tag]+1
            else:
                tags_counter[tag] = 1

    allowed_tags = tags_counter.keys()


    ####Contagem de etiquetas na base de treino
    word_freq = {}
    for s in phrases_train_rare:
        for tk in s:
            if word_freq.get(tk[0]) == None:
                word_freq[tk[0]] = Counter()
            word_freq[tk[0]][tk[1]] = word_freq[tk[0]][tk[1]] + 1

    return word_freq, word_counts, tags_counter, allowed_tags



def get_emission_probs(word_freq, tags_counter):
    e = {}
    for word in word_freq.keys():
        for tag in tags_counter.keys():
            if tags_counter[tag] == 0:
                print(tag)
            e[(word,tag)] = float((word_freq[word].get(tag,smooth_emission_not_seen)+1)/float(tags_counter[tag]))
    return e
#e_prob = get_emission_probs(word_freq, tags_counter)    


def prep_sentence_start_stop(phrase):
    return ['*'] + ['*'] + [word[1] for word in phrase] + ['STOP']
    
def get_transition_probs(phrases,word_feq,tags_counter):

    bigrams_c = Counter()
    for phrase in phrases:
        bigrams = nltk.bigrams(prep_sentence_start_stop(phrase))
        for bigram in bigrams:
            bigrams_c[bigram] +=1

    trigrams_c = Counter()
    for phrase in phrases:
        trigrams = nltk.trigrams(prep_sentence_start_stop(phrase))
        for trigram in trigrams:
            trigrams_c[trigram] +=1

    trigrams_p = {}
    for trigram, trigram_count in trigrams_c.items():
        bigram = (trigram[0],trigram[1])
        bigram_count = bigrams_c.get(bigram)
        if bigram_count:
            trigrams_p[trigram] = float(trigram_count)/float(bigram_count)
        else:
            trigrams_p[trigram] = 0
        
    return trigrams_p
    
    
#q_prob = get_transition_probs(phrases,word_freq,tags_counter)


def prep_sentence(s, word_counts):
    return [tk[0] if word_counts.get(tk[0]) != None else RARE_WORD for tk in s]

def viterbi(sents, e_prob,q_prob,allowed_tags):

    
    def S(k):
        if k == -1 or k == 0:
            return ['*']
        else:
            return allowed_tags
    pi = {}
    pi[(0, '*','*')] = 1

    bp = {}
    

    tagged = []
    for sent in sents:
        for k in range(1, len(sent)+1):
            for u in S(k-1):
                for v in S(k):
                    max_p = -1
                    max_tag = None
                    for w in S(k-2):
                        prob = pi[(k-1,w,u)] * q_prob.get((w,u,v),0) * e_prob[(sent[k-1],v)]
                        if (prob > max_p):
                            max_p = prob
                            max_tag = w
                    pi[(k,u,v)] = max_p
                    bp[(k,u,v)] = max_tag
        

        max_p = -1
        max_u_tag = None
        max_v_tag = None
        n = len(sent)
        for u in S(n-1):
            for v in S(n):
                prob = pi[(n,u,v)] * q_prob.get((u,v,'STOP'),0)
                if (prob > max_p):
                    max_p = prob
                    max_u_tag = u
                    max_v_tag = v

        tags = []
        tags.append(max_v_tag)
        tags.append(max_u_tag)
        for i,k in enumerate(range(n-2,0, -1)):
            tags.append(bp[(k+2, tags[i+1], tags[i])])

        tags = list(reversed(tags))

        tagged_sentence = []
        for j in range(0, n):
            tagged_sentence.append((sent[j], tags[j]))
        
        tagged.append(tagged_sentence)
    
    return tagged




def evaluate_viterbi(word_freq, word_counts, allowed_tags, e_prob, q_prob, phrases_test,sample=-1, log_fn=None):
    start_time = timeit.default_timer()
    
    corretas_baseline = 0
    corretas_viterbi = 0
    totais = 0
    
    
    if sample != -1:
        testwith = sample
    else:
        testwith = len(phrases_test)


    if log_fn:
        print('Total of '+str(testwith)+' phrases to run.')
        log_fn('Total of '+str(testwith)+' phrases to run.')
        
    counter = 0
    for s in phrases_test[:testwith]:

        ##Logging...
        counter+=1
        if log_fn and counter%50000 == 0:
            print(str(counter)+' out of '+str(testwith)+' ('+str(int(100*counter/testwith))+'%)')
            log_fn(str(counter)+' out of '+str(testwith)+' ('+str(int(100*counter/testwith))+'%)')


        ## Algorithm...
        tagged = viterbi([prep_sentence(s,word_counts)], e_prob, q_prob, allowed_tags)[0]
        for tk_golden, tk_pred in zip(s,tagged):
            totais+=1
            if tk_golden[1] == word_freq[tk_pred[0]].most_common(1)[0][0]:
                corretas_baseline+=1
            if tk_golden[1] == tk_pred[1]:
                corretas_viterbi+=1

    elapsed = timeit.default_timer() - start_time
    
    return corretas_viterbi, corretas_baseline, totais, elapsed