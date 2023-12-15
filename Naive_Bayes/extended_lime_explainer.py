from lime.lime_text import (
    LimeTextExplainer,
    IndexedString,
    TextDomainMapper,
    IndexedCharacters,
    explanation
)
from lime.explanation import Explanation
import sklearn
import numpy as np
import scipy as sp
from lime import lime_base
from sklearn.utils import check_random_state
from functools import partial
import re



class ExtendedLimeTextExplainer(LimeTextExplainer):
    def __init__(self,
                kernel_width=25,
                kernel=None,
                verbose=False,
                class_names=None,
                feature_selection='auto',
                split_expression=r'\W+',
                bow=True,
                mask_string=None,
                random_state=None,
                char_level=False):
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                        random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

    def explain_instance(
        self,
        text_instance,
        classifier_fn,
        labels=(1,),
        top_labels=None,
        num_features=10,
        num_samples=5000,
        distance_metric="cosine",
        model_regressor=None,
        exception_words=None):
        self.exception_words = exception_words

        indexed_string = (IndexedCharacters(
            text_instance, bow=self.bow, mask_string=self.mask_string)
                          if self.char_level else
                          IndexedString(text_instance, bow=self.bow,
                                        split_expression=self.split_expression,
                                        mask_string=self.mask_string))
        domain_mapper = TextDomainMapper(indexed_string)

        data, yss, distances = self.__data_labels_distances(
            indexed_string, classifier_fn, num_samples,
            distance_metric=distance_metric)
       
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
       
        ret_exp.predict_proba = yss[0]
      
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        
     
        for label in labels:
           
            ret_exp.score= {}
            ret_exp.local_pred = {}
            

            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
    
        return ret_exp
        """  self.explain_instance(
        text_instance,
        classifier_fn=classifier_fn,
        labels=labels,
        top_labels=top_labels,
        num_features=num_features,
        num_samples=num_samples,
        distance_metric=distance_metric,
        model_regressor=model_regressor) """

    def __data_labels_distances(self,
                                indexed_string,
                                classifier_fn,
                                num_samples,
                                distance_metric="cosine"):

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x,
                x[0],
                metric=distance_metric
              ).ravel() * 100

        doc_size = indexed_string.num_words()
       
        """ sample means the amount of words that will change.
        it generates a vector of size num_samples in which ieach element contain a random number of words that will be deleted
        if sample[0] = 2 => 2 words will be deleted in the first sentence
        however, if doc_size = 2 none words can be changed, so all 
        new sentences are the same, therefore, it is created a vector of size num_samples with the same sentence, containing only the exception words.
        num_samples means how many new sentences will be created (default =5000) """
        
        inverse_data = [indexed_string.raw_string()]
        data = np.ones((num_samples, doc_size)) 
        data[0] = np.ones(doc_size)
       
        # update exception words position since lime split and counts only words, not punctuation
        # lime also remove duplicated words, leaving only one word each type
        words_with_punct =  inverse_data[0].split()
        
        cleaned_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', inverse_data[0])
        words_clean = cleaned_sentence.split()

        unique_words_set = set()
        unique_words_list = []

        for word in words_clean:
            if word not in unique_words_set:
                unique_words_set.add(word)
                unique_words_list.append(word)

        exception_word1_old = words_with_punct[self.exception_words[0]]
        exception_word2_old = words_with_punct[self.exception_words[1]]

        exception_word1_new_position = unique_words_list.index(exception_word1_old)
        exception_word2_new_position = unique_words_list.index(exception_word2_old)

        new_exception_words_position = (exception_word1_new_position, exception_word2_new_position )

        if doc_size >= 3:
            sample = self.random_state.randint(1, doc_size - 1, num_samples - 1)    
            features_range = list(set(range(doc_size)) - set(new_exception_words_position))

            for i, size in enumerate(sample, start=1):

                inactive = self.random_state.choice(features_range, size, replace=False)
                data[i, inactive] = 0
                inverse_data.append(indexed_string.inverse_removing(inactive))

        
        else:
            for i in range(num_samples-1):
                inverse_data.append(indexed_string.raw_string())


        labels = classifier_fn(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances