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
#from lime_utils import class_names
from extended_lime_explainer import ExtendedLimeTextExplainer
#from pre_processing_stanza import pre_processing
#from lime_utils import evaluation_lime
from NB_LIME import explainer, class_labels, c


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

# for testing, to speed
rates = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]


#one different random seed
seeds = seeds = [1, 28989]


def rbo_per_array( arrays):
    num_arrays = len(arrays[1])
    rbo_per_sampling_rate = []
    rbo_per_seed =[]

    for i in range(2):
        base = arrays[i][4]
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
    plt.title('Inherent Stability - Naive Bayes')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig('output_plot.png')
    plt.show()

def rate_succeful_attackss(samples):
    
    fail_attack = 0
    succeful_attack = 0
    RBO_total = []
    for index, sample in samples.iterrows():
    #for sample in samples:
    # try:
        RBO_sim = Stability(sample).similarity_attack()
        #print(RBO_sim)
        #except:
        #   continue
        if RBO_sim > 0.5:
            fail_attack += 1
        else:
            succeful_attack += 1
        RBO_total.append(RBO_sim)

    final_rate = succeful_attack / (succeful_attack + fail_attack)

    return final_rate, RBO_total

def rate_succeful_attackss(samples):
    i = 0
    fail_attack = 0
    succeful_attack = 0
    RBO_total = []
    errors = []
    index_ = []
    RBO_total_backup = []
    for index, sample in samples.iterrows():

        try:
            RBO_sim = Stability(sample).similarity_attack(rate= 2000)
            #print(RBO_sim)
            #except:
            #   continue
            # Save to CSV every 5 iterations
            RBO_total.append(RBO_sim)
            RBO_total_backup.append(RBO_sim)

            if RBO_sim > 0.5:
                fail_attack += 1
            else:
                succeful_attack += 1

            if (i + 1) % 5 == 0 or i == len(samples) - 1:
                data = {'RBO_sim': RBO_total}
                df = pd.DataFrame(data)
                df.to_csv("RBO_sim.csv", mode='a', header=False, index=False)

                RBO_total = []
            i = i +1

        except Exception as e:
            print(f"######## Skipping sentence: {sample}. Error: {e}")
            index_.append(i)
            errors.append(e)
            errors_df = pd.DataFrame({"index":index_,
                                        "error":errors}) 
            errors_df.to_csv("errors_stability_parameter.csv", mode='a')
            
    final_rate = succeful_attack / (succeful_attack + fail_attack)
    
    
    return final_rate, RBO_total_backup

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
                explainer = ExtendedLimeTextExplainer(class_names=class_labels, random_state= seed)       
                exp = explainer.explain_instance(self.sentence, c.predict_proba, num_features=number_of_tokens, num_samples=rate, exception_words=(self.subj_entity_word, self.obj_entity_word))
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
        words = sentence.split(" ")
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

    def substitute_words_stability(self,sentence, important_words):
        words = word_tokenize(sentence)
        
        
        lowest_similarity= 1.0  # Initialize the lowest stability as 1.0 (maximum)
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

        exp = explainer.explain_instance(self.sentence, c.predict_proba, num_features=num_feat, num_samples=rate, exception_words=(self.subj_entity_word, self.obj_entity_word))
        important_words_base = self.get_most_important_words(self.sentence, exp)
        perturbed_sentence = self.substitute_words_stability(self.sentence, important_words_base)
        #print("pertubed sentence: ", perturbed_sentence)

        splitted_pert_sentence = perturbed_sentence.split()
        exp_p = explainer.explain_instance(perturbed_sentence, c.predict_proba, num_features=num_feat,  num_samples=rate, exception_words=(self.subj_entity_word, self.obj_entity_word))
        
        # important words is actually the result of the explanation from LIME
        #for both (pertubed and not pertubed the entities are added)
        important_words_p = self.get_most_important_words(perturbed_sentence, exp_p)

        # both k features of explanation are compared with RBO
        # if rbo < 0.5, attack succeful, later compute rate of succeful attacks per total attacks
        #p=0.8 as defined in the paper
        RBO_similarity = rbo.RankingSimilarity(important_words_base, important_words_p).rbo(p=0.8)

        return RBO_similarity

 