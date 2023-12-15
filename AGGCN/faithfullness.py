import math
import random
import numpy as np
import lime
from lime_utils import evaluation_lime
from pre_processing_stanza import pre_processing
from extended_lime_explainer import ExtendedLimeTextExplainer

class_names= ['Other', 'Entity-Destination', 'Cause-Effect', 'Member-Collection', 'Entity-Origin', 'Message-Topic', 'Component-Whole', 'Instrument-Agency', 'Product-Producer', 'Content-Container']

#new explainer that dont remove the relations from the sentence when creating samples
explainer = ExtendedLimeTextExplainer(class_names= class_names)


class Faithfullness:
    def __init__(self, sample):
        self.subj_start = sample.subj_start
        self.obj_start = sample.obj_start
        self.sentence = sample.sentence
        self.token = sample.token
        self.subj_entity_word = self.token[self.subj_start]
        self.obj_entity_word = self.token[self.obj_start]


        
    def remove_words(self,sentence, words_to_remove, subj_start, obj_start):
        words = sentence.split()

        exception_words = [words[subj_start], words[obj_start]]
        words_to_remove = [word for word, _ in words_to_remove]
        
        words_to_remove_updated = [word for word in words_to_remove if word not in exception_words]

        filtered_words = [word for word in words if word.lower() not in words_to_remove_updated]
        new_subj_start = filtered_words.index(exception_words[0])
        new_obj_start = filtered_words.index(exception_words[1])


        new_sentence = ' '.join(filtered_words)

        return new_sentence, new_subj_start, new_obj_start
    ## ---- comprehensiveness

    def comprehesiveness(self, num_features =5, num_samples_lime = 50):
        
      
        exp = explainer.explain_instance(text_instance=self.sentence, classifier_fn=self.lime_pipeline_wrapper, num_features=num_features, num_samples = num_samples_lime,exception_words=(self.subj_start, self.obj_start))
        probability = max(exp.predict_proba)
        label = np.where(exp.predict_proba == probability)
       
        #create new sentence without the rationals
        words_to_remove = exp.as_list()
        new_sentence, new_subj_start, new_obj_start = self.remove_words(self.sentence, words_to_remove, subj_start=self.subj_start, obj_start=self.obj_start)
        
        #get lenght of the new sentence to be used as num_features in explainer
        len_new_sentence = len(new_sentence.split())
        if num_features > len_new_sentence:
            num_features = len_new_sentence

        exp_new = explainer.explain_instance(text_instance=new_sentence, classifier_fn=self.lime_pipeline_wrapper, num_features=num_features, num_samples = num_samples_lime, exception_words=(new_subj_start, new_obj_start) )
        probability_new = exp_new.predict_proba[label] 

        compr = probability - probability_new

        return compr


    def comprehensiveness_aopc(self, num_samples_lime = 10):
        # get bins of percentage of tokens used to calculate suff
        bins = [0.01, 0.05, 0.1, 0.2, 0.5]
        #bins = [ 0.2, 0.5, 0.7, .9]

        total_number_of_tokens = len(self.sentence.split())

        total_compr =[]

        for i in bins:
            number_of_tokens = math.ceil(total_number_of_tokens*i)

            compr = self.comprehesiveness(num_features=number_of_tokens, num_samples_lime=num_samples_lime)

            total_compr.append(compr)

        aopc_compr = sum(total_compr)/len(bins)

        return aopc_compr


    def comprehensiveness_aopc_random(self, num_samples_lime = 10):
    # get bins of percentage of tokens used to calculate compr
        bins = [0.01, 0.05, 0.1, 0.2, 0.5] #as the paper
        #bins = [ 0.2, 0.5, 0.7, .9]

        sentence_lenght = len(self.token)
        splitted_setence = self.token

        total_compr =[]
        entities = [self.subj_entity_word, self.obj_entity_word]

        for i in bins:
            #round the percentage of words UP
            number_of_tokens = math.ceil(sentence_lenght*i)
            exp = explainer.explain_instance(self.sentence, classifier_fn=self.lime_pipeline_wrapper, num_features=number_of_tokens, num_samples = num_samples_lime, exception_words=(self.subj_start, self.obj_start))
            probability = max(exp.predict_proba)
            label = np.where(exp.predict_proba == probability)

            #create new sentence without the rationals
            words_to_remove = exp.as_list()
            num_tokens_to_remove =len(words_to_remove)
        

            #choose tokens to remove randomly
            tokens_to_remove = random.sample(splitted_setence,num_tokens_to_remove)
           
            tokens_to_remove_without_entities = [word for word in tokens_to_remove if word not in entities]
            #i still need to get the entities position
            #new_sentence, new_subj_start, new_obj_start = self.remove_words(self.sentence, tokens_to_remove, subj_start=self.subj_start, obj_start=self.obj_start)

            new_sentence = [word for word in splitted_setence if word not in tokens_to_remove_without_entities]

            new_subj_start = new_sentence.index(entities[0])
            new_obj_start = new_sentence.index(entities[1])
            len_new_sentence = len(new_sentence)
            if number_of_tokens > len_new_sentence:
                number_of_tokens = len_new_sentence
            
            #new_sentence = [word for word in splitted_setence if word not in tokens_to_remove and word not in entities]
            new_sentence  = ' '.join(new_sentence)

            exp_new = explainer.explain_instance(text_instance=new_sentence, classifier_fn=self.lime_pipeline_wrapper, num_features=number_of_tokens, num_samples = num_samples_lime, exception_words=(new_subj_start, new_obj_start))
            probability_new = exp_new.predict_proba[label] 

            compr = probability - probability_new

            total_compr.append(compr)

        aopc_suff_random = sum(total_compr)/len(bins)

        return aopc_suff_random

    #---------- suffiency

    def suffiency(self, num_features = 3, num_samples_lime = 50):


        exp = explainer.explain_instance(text_instance=self.sentence, classifier_fn=self.lime_pipeline_wrapper, num_features=num_features, num_samples = num_samples_lime,exception_words=(self.subj_start, self.obj_start))
        probability = max(exp.predict_proba)
        label = np.where(exp.predict_proba == probability)
        
        #create new sentence with only the rationals
        words_to_remain = exp.as_list()
        words_to_remain  = [word for word, _ in words_to_remain]

        indices = [self.token.index(word) for word in words_to_remain]
        indices_set = set(indices)
        exception_words_index = (self.subj_start, self.obj_start)
        indices_set.update(set(exception_words_index))
        updated_indices = sorted(indices_set)

        # Extract words from the sentence based on the updated indices
        new_sentence = [self.token[i] for i in updated_indices]
        len_new_sentence = len(new_sentence)

        if num_features > len_new_sentence:
            num_features = len_new_sentence

        # get new postion of the entities
        new_subj_start = new_sentence.index(self.subj_entity_word)
        new_obj_start = new_sentence.index(self.obj_entity_word)


        new_sentence  = ' '.join(new_sentence)
        exp_new = explainer.explain_instance(text_instance=new_sentence, classifier_fn=self.lime_pipeline_wrapper, num_features=num_features, num_samples = num_samples_lime, exception_words=(new_subj_start, new_obj_start) )
        probability_new = exp_new.predict_proba[label] 
        suff = probability - probability_new

        return suff


    def suffiency_aopc(self, num_samples_lime=50):
        # get bins of percentage of tokens used to calculate suff
        bins = [0.01, 0.05, 0.1, 0.2, 0.5] #as the paper
        #bins = [ 0.2, 0.5, 0.7, .9]

        #reduced form to speed time
        #bins = [0.6, 0.7, 0.8, 0.9] 

        total_number_of_tokens = len(self.token)

        total_suff =[]

        for i in bins:
            
            number_of_tokens = math.ceil(total_number_of_tokens*i)

            suff = self.suffiency(num_features=number_of_tokens, num_samples_lime=num_samples_lime)

            total_suff.append(suff)

        aopc_suff = sum(total_suff)/len(bins)

        return aopc_suff


    def suffiency_aopc_random(self, num_samples_lime = 50):
        """ Suffiency: sentence with only the rationals"""
        # get bins of percentage of tokens used to calculate suff
        #bins = [0.7, .8]
        bins = [0.01, 0.05, 0.1, 0.2, 0.5] #as the paper


        sentence_lenght = len(self.token)

        splitted_setence = self.token

        total_suff =[]

        entities = [self.subj_entity_word, self.obj_entity_word]


        for i in bins:
            #round the percentage of words UP
            number_of_tokens = math.ceil(sentence_lenght*i)
            exp = explainer.explain_instance(self.sentence, classifier_fn=self.lime_pipeline_wrapper, num_features=number_of_tokens, num_samples = num_samples_lime, exception_words=(self.subj_start, self.obj_start))
            probability = max(exp.predict_proba)
            label = np.where(exp.predict_proba == probability)

            tokens = exp.as_list()
            tokens  = [word for word, _ in tokens]
        
            # I have a list of words -> take randomly x words to calculate againthe prediction probability
            # x is the parameter number_of_tokens
            # number of words that are not rationals
            num_tokens_to_remove = sentence_lenght - number_of_tokens

            #choose tokens to remove randomly
            tokens_to_remove = random.sample(splitted_setence,num_tokens_to_remove)

            tokens_to_remove_without_entities = [word for word in tokens_to_remove if word not in entities]
            
            new_sentence = [word for word in splitted_setence if word not in tokens_to_remove_without_entities]
            len_new_sentence = len(new_sentence)
            if number_of_tokens > len_new_sentence:
                number_of_tokens = len_new_sentence

            new_subj_start = new_sentence.index(entities[0])
            new_obj_start = new_sentence.index(entities[1])

            new_sentence  = ' '.join(new_sentence)

            exp_new = explainer.explain_instance(text_instance=new_sentence, classifier_fn=self.lime_pipeline_wrapper, num_features=number_of_tokens, num_samples = num_samples_lime, exception_words=(new_subj_start, new_obj_start))
            probability_new = exp_new.predict_proba[label] 

            suff = probability - probability_new

            total_suff.append(suff)

        aopc_suff_random = sum(total_suff)/len(bins)

        return aopc_suff_random

    def lime_pipeline_wrapper(self,texts):
        return self.lime_pipeline(texts)
    
    def lime_pipeline(self, texts):     

        subj_entity_word = self.subj_entity_word
        obj_entity_word =self.obj_entity_word
        number_of_texts = len(texts)

        # creates json file in 'dataset/semeval/pre_processed_data.json'
        pre_processed_sample = pre_processing(texts=texts, relation="Cause-Effect", subj_entity_word=subj_entity_word, obj_entity_word=obj_entity_word)

        # uses the new created file
        # probability, label = evaluation_lime(dataset_='pre_processed_data_')
        probability, label = evaluation_lime(dataset_='pre_processed_data_')

        reshaped_probs = np.array(probability).reshape(number_of_texts, 10)

        return reshaped_probs