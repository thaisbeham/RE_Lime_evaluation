import tensorflow as tf
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fse import Vectors
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import rbo
import numpy as np
import lime
#from lime.lime_text import LimeTextExplainer
from lime_utils import class_names
from extended_lime_explainer import ExtendedLimeTextExplainer
from pre_processing_stanza import pre_processing
from lime_utils import evaluation_lime


#nltk.download('stopwords')
#nltk.download('punkt')

##------ parameters
doc_similarity = 0.5
num_feat = 4
##-----------------


stop_words = set(stopwords.words('english'))
vecs_ws = Vectors.from_pretrained("paragram-300-ws353")
model = tf.saved_model.load('../tf_hub_model/')

### ------- inherent similarity

# as the paper
#rates =[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ]
rates = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]



# for testing, to speed
#rates = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]

#one different random seed
seeds = [1, 28989]
explainer = ExtendedLimeTextExplainer(class_names=class_names)

def rbo_per_array( arrays):
    num_arrays = len(arrays[1])
    rbo_per_sampling_rate = []
    rbo_per_seed =[]

    for i in range(2):
        base = arrays[i][9]
        for j in range(num_arrays):
            rbo_value = rbo.RankingSimilarity(arrays[i][j], base).rbo(p=0.8)
            rbo_per_sampling_rate.append(rbo_value)
        rbo_per_seed.append(rbo_per_sampling_rate)
        rbo_per_sampling_rate = []
    
    rbo_data = {'seed1': rbo_per_seed[0],
                'seed2': rbo_per_seed[1]}

    return rbo_data


def mean_all_inhe_stability(samples):
    all_data_seed1 = []
    all_data_seed2 = []
    total_arrays = []

    for index, sample in samples.iterrows():
        try:
            arrays = Stability(sample).change_sampling_rate_seed()
            inherent_similarity_data = rbo_per_array(arrays)
            all_data_seed1.append(inherent_similarity_data['seed1'])
            all_data_seed2.append(inherent_similarity_data['seed2'])
            total_arrays.append(arrays)
        except Exception as e:
            print(f"Error processing sample {index}: {e}")

    mean_seed1 = np.mean(all_data_seed1, axis=0) 
    mean_seed2 = np.mean(all_data_seed2, axis=0)

    return mean_seed1, mean_seed2, total_arrays

def plot_grap_stability_inhe(x, y1, y2):

    # Create the first scatterplot with transparency 0.8 and trendline
    plt.scatter(x, y1, label='Seed1', alpha=0.8)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y1, 1))(np.unique(x)), color='blue', linestyle='--')

    # Create the second scatterplot with a different transparency level and trendline
    plt.scatter(x, y2, label='Seed2', alpha=0.8)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y2, 1))(np.unique(x)), color='orange', linestyle='--')


    # Set axis limits to start at zero
    plt.xlim(0, max(x)+ 500)
    plt.ylim(0, max(max(y1), max(y2)) + 0.2)


    # Customize the plot (labels, title, legend, etc.)
    plt.xlabel('Sampling rates')
    plt.ylabel('RBO (p=0.8)')
    plt.title('Inherent Stability - AGGCN')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig('output_plot.png')
    plt.show()

def rate_succeful_attackss(samples):
    
    fail_attack = 0
    succeful_attack = 0
    for index, sample in samples.iterrows():
    # try:
        RBO_sim = Stability(sample).similarity_attack()
        #print(RBO_sim)
        #except:
        #   continue
        if RBO_sim > 0.5:
            fail_attack += 1
        else:
            succeful_attack += 1

    final_rate = succeful_attack / (succeful_attack + fail_attack)

    return final_rate
class Stability:
    def __init__(self, sample):
        self.subj_start = sample.subj_start
        self.obj_start = sample.obj_start
        self.sentence = sample.sentence
        self.token = sample.token
        self.subj_entity_word = self.token[self.subj_start]
        self.obj_entity_word = self.token[self.obj_start]

    def change_sampling_rate_seed(self, rates = rates, number_of_tokens = 5, seeds = seeds):
        
        words_per_sampling_rate =[]
        words_per_seed =[]
        for seed in seeds:

            for rate in rates:
                #print(seed)
                explainer = ExtendedLimeTextExplainer(class_names=class_names, random_state= seed)       
                exp = explainer.explain_instance(self.sentence, self.lime_pipeline, num_features=number_of_tokens, num_samples=rate, exception_words=(self.subj_entity_word , self.obj_entity_word))
                most_important_words = self.get_most_important_words(self.sentence, exp)
                words_per_sampling_rate.append(most_important_words)

            words_per_seed.append(words_per_sampling_rate)
            words_per_sampling_rate = []

        return words_per_seed

    ### ------- parameter similarity
    def embed(self, input):
        return model(input)

    def get_most_important_words(self,sentence, exp):
        exp_list = exp.as_map()[1]
        exp_list = sorted(exp_list, key=lambda x: x[1], reverse= True)
        #exp_weight = [x[1] for x in exp_list]
        #print("exp_list: ", exp_list, len(exp_list))
        #words = sentence.split(" ")
        words = re.findall(r'\b\w+\b', sentence)
        # print("words: ", words, len(words))
        #word_weight_mapping = [(words[word_id+1],weight) for word_id, weight in exp_list]
        #why I need to add +1 in [(words[word_id+1],weight) ??
        word_weight_mapping_dict = { words[word_id]: weight for word_id, weight in exp_list}
        
        # get only keys
        vector_words = list(word_weight_mapping_dict.keys())

        #add entities
        vector_words.append(self.subj_entity_word)
        vector_words.append(self.obj_entity_word)

        vector_words = np.unique(vector_words)

        return vector_words

    def get_similarity(self,sentence1, sentence2):

        messages = [sentence1, sentence2]

        message_embeddings = self.embed(messages)

        corr = np.inner(message_embeddings[0], message_embeddings[1])

        return  corr

    def substitute_words_similarity(self,sentence, important_words):
        words = word_tokenize(sentence)
        
        
        lowest_similarity = 1.0  # Initialize the lowest similarity as 1.0 (maximum)
        modified_sentence = sentence
        consecutive_iterations = 0  # Counter for consecutive iterations without improvement
        modified_words = set()  # Set to keep track of modified words
        
        while lowest_similarity > doc_similarity:
            # Create a copy of the sentence to work with
            new_sentence = modified_sentence
            
            # Choose a random word to substitute, ensuring it's not a stopword, an important word, or already modified
            eligible_words = [word for word in words if word.lower() not in stop_words and word not in important_words and word not in modified_words]
            if not eligible_words:
                # If all words have been modified, break the loop
                break
            
            word_to_substitute = random.choice(eligible_words)
            
            # Use NLTK to determine the part of speech of the word
            word_pos = nltk.pos_tag([word_to_substitute])[0][1]
            
            # Find the most similar word with the same part of speech using the word vector model
            try:
                neighbors = vecs_ws.most_similar(word_to_substitute)
                substitute_word = neighbors[0][0]
                
                # Check the part of speech of the substitute word
                substitute_pos = nltk.pos_tag([substitute_word])[0][1]
                
                # Allow replacement of nouns with verbs
                if word_pos == substitute_pos or (word_pos.startswith("NN") and substitute_pos.startswith("VB")):
                    # Substitute the chosen word with the most similar word in the copy of the sentence
                    new_sentence = new_sentence.replace(word_to_substitute, substitute_word)
                    
                    # Add the modified word to the set of modified words
                    modified_words.add(word_to_substitute)
            except KeyError:
                pass
            
            # Calculate the similarity between the original sentence and the new sentence
            similarity = round(self.get_similarity(sentence, new_sentence), 2)  # Round to two decimal places
            
            if similarity < lowest_similarity:
                lowest_similarity = similarity
                modified_sentence = new_sentence
                consecutive_iterations = 0  # Reset the counter
            else:
                consecutive_iterations += 1
                
            if consecutive_iterations >= 5:
                break
                
        return modified_sentence

    def similarity_attack(self, rate = 10):
        # sampling rate 'num_samples' set to 10 for speed propose

        exp = explainer.explain_instance(self.sentence, self.lime_pipeline, num_features=num_feat, num_samples=rate, exception_words=(self.subj_entity_word, self.obj_entity_word))
        important_words_base = self.get_most_important_words(self.sentence, exp)
        print("important words: ", important_words_base)
        perturbed_sentence = self.substitute_words_similarity(self.sentence, important_words_base)
        print("pertubed sentence: ", perturbed_sentence)

        splitted_pert_sentence = perturbed_sentence.split()
        #splitted_pert_sentence = re.findall(r'\b\w+\b', perturbed_sentence)
        #splitted_pert_sentence = re.split(r'\s+|-|/|\'|\.|\,', perturbed_sentence)
        print("splitted pertubed sentence: ", splitted_pert_sentence)
        new_subj_start = splitted_pert_sentence.index(self.subj_entity_word)
        print("new subj start: ", new_subj_start)
        new_obj_start = splitted_pert_sentence.index(self.obj_entity_word)
        print("new obj start: ", new_obj_start)
        exp_p = explainer.explain_instance(perturbed_sentence, self.lime_pipeline, num_features=num_feat,  num_samples=rate, exception_words=(self.subj_entity_word, self.obj_entity_word))
        print("exp_p: ",exp_p.as_map()[1])
        # important words is actually the result of the explanation from LIME
        #for both (pertubed and not pertubed the entities are added)
        important_words_p = self.get_most_important_words(perturbed_sentence, exp_p)
        print("important words perturbed sentence: ", important_words_p)
        # both k features of explanation are compared with RBO
        # if rbo < 0.5, attack succeful, later compute rate of succeful attacks per total attacks
        #p=0.8 as defined in the paper
        RBO_similarity = rbo.RankingSimilarity(important_words_base, important_words_p).rbo(p=0.8)

        return RBO_similarity

 
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
    