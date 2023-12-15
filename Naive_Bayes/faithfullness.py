from NB_LIME import explainer_c
import math
import random
import numpy as np
from extended_lime_explainer import ExtendedLimeTextExplainer

explainer, c = explainer_c()

def remove_words(sentence, words_to_remove, subj_start, obj_start):
    words = sentence.split()

    exception_words = [words[subj_start], words[obj_start]]
    words_to_remove = [word for word, _ in words_to_remove]
    
    words_to_remove_updated = [word for word in words_to_remove if word not in exception_words]

    filtered_words = [word for word in words if word.lower() not in words_to_remove_updated]
    new_subj_start = filtered_words.index(exception_words[0])
    new_obj_start = filtered_words.index(exception_words[1])


    new_sentence = ' '.join(filtered_words)

    return new_sentence, new_subj_start, new_obj_start

def comprehesiveness(df, num_samples= 5000, num_features =5):
    exp = explainer.explain_instance(df.sentence, c.predict_proba, num_features=num_features, top_labels=2, num_samples=num_samples, exception_words=(df.subj_start, df.obj_start) )
    probability = max(exp.predict_proba)
    label = np.where(exp.predict_proba == probability)
    
    #create new sentence without the rationals
    #also here the exception words can not be removed
    words_to_remove = exp.as_list()
    new_sentence, new_subj_start, new_obj_start = remove_words(df.sentence, words_to_remove, df.subj_start, df.obj_start)    

    len_new_sentence = len(new_sentence.split())
     
    exp_new = explainer.explain_instance(new_sentence, c.predict_proba,  num_features=len_new_sentence, top_labels=2, num_samples=num_samples, exception_words=(new_subj_start, new_obj_start))
    probability_new = exp_new.predict_proba[label] 

    compr = probability - probability_new

    return compr

def comprehensiveness_aopc(df, num_samples= 5000, num_features =5):
    # get bins of percentage of tokens used to calculate suff
    #bins = [0.01, 0.05, 0.1, 0.2, 0.5]
    bins = [ 0.2, 0.5, 0.7, .9]

    total_number_of_tokens = len(df.token)

    total_compr =[]

    for i in bins:
        number_of_tokens = math.ceil(total_number_of_tokens*i)

        exp = explainer.explain_instance(df.sentence, c.predict_proba, num_features=number_of_tokens, top_labels=2, num_samples=num_samples, exception_words=(df.subj_start, df.obj_start) )
        probability = max(exp.predict_proba)
        label = np.where(exp.predict_proba == probability)

        #create new sentence without the rationals
        words_to_remove = exp.as_list()
        new_sentence, new_subj_start, new_obj_start = remove_words(df.sentence, words_to_remove, df.subj_start, df.obj_start)    

        len_new_sentence = len(new_sentence)
        exp_new = explainer.explain_instance(new_sentence, c.predict_proba,  num_features=len_new_sentence, top_labels=2, num_samples=num_samples, exception_words=(new_subj_start, new_obj_start))
        probability_new = exp_new.predict_proba[label] 

        suff = probability - probability_new

        total_compr.append(suff)

    aopc_compr = sum(total_compr)/len(bins)

    return aopc_compr

def comprehensiveness_aopc_random(df, num_samples= 5000, num_features =5):
# get bins of percentage of tokens used to calculate compr
    #bins = [0.01, 0.05, 0.1, 0.2, 0.5] #as the paper
    bins = [ 0.2, 0.5, 0.7, .9]
    sentence = df.sentence
    sentence_lenght = len(sentence.split())

    splitted_sentence = sentence.split()

    exception_words = [splitted_sentence[df.subj_start], splitted_sentence[df.obj_start]]

    total_compr =[]

    for i in bins:
        #round the percentage of words UP
        number_of_tokens = math.ceil(sentence_lenght*i)
        #print("choosen number of features for the explainer: ", number_of_tokens)
        exp = explainer.explain_instance(sentence, c.predict_proba,  num_features=number_of_tokens, top_labels=2, num_samples=num_samples, exception_words=(df.subj_start, df.obj_start))
        probability = max(exp.predict_proba)
        label = np.where(exp.predict_proba == probability)

        #create new sentence without the rationals
        words_to_remove = exp.as_list()
        num_tokens_to_remove =len(words_to_remove)

        #choose tokens to remove randomly
        tokens_to_remove = random.sample(splitted_sentence,num_tokens_to_remove)

        #ensure that exception words are not removed
        tokens_to_remove_updated = [word for word in tokens_to_remove if word not in exception_words]
        new_sentence = [word for word in splitted_sentence if word not in tokens_to_remove_updated]

        new_subj_start = new_sentence.index(exception_words[0])
        new_obj_start = new_sentence.index(exception_words[1])
        new_sentence  = ' '.join(new_sentence)

        len_new_sentence = len(new_sentence)

        exp_new = explainer.explain_instance(new_sentence, c.predict_proba, num_features=len_new_sentence, top_labels=2, num_samples=num_samples, exception_words=(new_subj_start, new_obj_start))
        probability_new = exp_new.predict_proba[label] 

        suff = probability - probability_new

        total_compr.append(suff)

    aopc_suff_random = sum(total_compr)/len(bins)

    return aopc_suff_random

### suffiency ----------------------------------------------

def suffiency(df, num_samples= 5000, num_features =5):
    sentence = df.sentence
    splitted_sentence = sentence.split()
    exception_words_index = (df.subj_start, df.obj_start)
    exception_words = [splitted_sentence[df.subj_start], splitted_sentence[df.obj_start]]
    
    exp = explainer.explain_instance(sentence, c.predict_proba, num_features=num_features, top_labels=2, num_samples=num_samples, exception_words=(df.subj_start, df.obj_start))
    probability = max(exp.predict_proba)
    label = np.where(exp.predict_proba == probability)

    
    #create new sentence with only the rationals
    words_to_remain = exp.as_list()
    # if the exception words are not here, add them
    words_to_remain  = [word for word, _ in words_to_remain]    

    indices = [splitted_sentence.index(word) for word in words_to_remain]
    indices_set = set(indices)
    indices_set.update(set(exception_words_index))
    updated_indices = sorted(indices_set)

    # Extract words from the sentence based on the updated indices
    new_sentence = [splitted_sentence[i] for i in updated_indices]
    len_new_sentence = len(new_sentence)

    new_subj_start = new_sentence.index(exception_words[0])
    new_obj_start = new_sentence.index(exception_words[1])
    
    new_sentence = " ".join(new_sentence)
    exp_new = explainer.explain_instance(new_sentence,  c.predict_proba, num_features=len_new_sentence, top_labels=2, num_samples=num_samples, exception_words=(new_subj_start, new_obj_start))
    probability_new = exp_new.predict_proba[label] 

    suff = probability - probability_new

    return suff

def suffiency_aopc(df, num_samples= 5000):
    # get bins of percentage of tokens used to calculate suff
    bins = [0.01, 0.05, 0.1, 0.2, 0.5] #as the paper
    #bins = [ 0.2, 0.5, 0.7, .9]

    sentence = df.sentence
    splitted_sentence =sentence.split()
    exception_words_index = (df.subj_start, df.obj_start)
    exception_words = [splitted_sentence[df.subj_start], splitted_sentence[df.obj_start]]
    
    total_number_of_tokens = len(sentence.split())

    total_suff =[]

    for i in bins:
        number_of_tokens = math.ceil(total_number_of_tokens*i)
        #print(number_of_tokens)
        exp = explainer.explain_instance(sentence, c.predict_proba, num_features=number_of_tokens, top_labels=2, num_samples=num_samples, exception_words=(df.subj_start, df.obj_start))
        probability = max(exp.predict_proba)
        label = np.where(exp.predict_proba == probability)

        #create new sentence with only the rationals
        words_to_remain = exp.as_list()
        words_to_remain  = [word for word, _ in words_to_remain]

        #ensure that the exception words are not removed
        indices = [splitted_sentence.index(word) for word in words_to_remain]
        indices_set = set(indices)
        indices_set.update(set(exception_words_index))
        updated_indices = sorted(indices_set)

        # Extract words from the sentence based on the updated indices
        new_sentence = [splitted_sentence[i] for i in updated_indices]

        new_subj_start = new_sentence.index(exception_words[0])
        new_obj_start = new_sentence.index(exception_words[1])
        len_new_sentence = len(new_sentence)

        new_sentence = " ".join(new_sentence)

        exp_new = explainer.explain_instance(new_sentence, c.predict_proba, num_features=len_new_sentence, top_labels=2, num_samples=num_samples, exception_words=(new_subj_start, new_obj_start))
        probability_new = exp_new.predict_proba[label] 

        suff = probability - probability_new

        total_suff.append(suff)

    aopc_suff = sum(total_suff)/len(bins)

    return aopc_suff


def suffiency_aopc_random(df, num_samples= 5000):
# get bins of percentage of tokens used to calculate suff
    bins = [ 0.2, 0.5, 0.7, .9]
    
    sentence = df.sentence
    splitted_sentence =sentence.split()
    sentence_lenght = len(splitted_sentence)

    exception_words = [splitted_sentence[df.subj_start], splitted_sentence[df.obj_start]]
    total_suff =[]

    for i in bins:
        #round the percentage of words UP
        number_of_tokens = math.ceil(sentence_lenght*i)
        #print("choosen number of features for the explainer: ", number_of_tokens)
        exp = explainer.explain_instance(sentence,  c.predict_proba, num_features=number_of_tokens, top_labels=2, num_samples=num_samples, exception_words=(df.subj_start, df.obj_start))
        probability = max(exp.predict_proba)
        label = np.where(exp.predict_proba == probability)

        #new sentence with only rationals
        tokens = exp.as_list()
        tokens  = [word for word, _ in tokens]
        #print("tokens selected as rationals: ", tokens)

        # I have a list of words -> take randomly x words to calculate againthe prediction probability
        # x is the parameter number_of_tokens
        num_tokens_to_remove = sentence_lenght - number_of_tokens

        #choose tokens to remove randomly
        tokens_to_remove = random.sample(splitted_sentence,num_tokens_to_remove)
        tokens_to_remove_updated = [word for word in tokens_to_remove if word not in exception_words]

        new_sentence = [word for word in splitted_sentence if word not in tokens_to_remove_updated]
        
        new_subj_start = new_sentence.index(exception_words[0])
        new_obj_start = new_sentence.index(exception_words[1])

        len_new_sentence = len(new_sentence)
        
        new_sentence  = ' '.join(new_sentence)

        exp_new = explainer.explain_instance(new_sentence,  c.predict_proba,num_features=len_new_sentence, top_labels=2, num_samples=num_samples, exception_words=(new_subj_start, new_obj_start))
        probability_new = exp_new.predict_proba[label] 

        suff = probability - probability_new

        total_suff.append(suff)

    aopc_suff_random = sum(total_suff)/len(bins)

    return aopc_suff_random

