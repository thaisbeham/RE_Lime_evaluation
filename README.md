# Evaluating LIME-based explanations of Relation Extraction models

Relation Extraction (RE) is a NLP field that intends to predict the relationship of entities. LIME, a popular and powerful explanation method, by definition, cannot handle explanations of RE tasks, since it arbitrarily removes words from a sentence, generating 5000 new samples, which results in removing the entities of the relation in many of them, breaking the pipeline when LIME tries to get the prediction of these new sentences. 

This project proposes a solution for this problem by modifying the class LimeTExtExplainer(), allowing it to perform all the functions of the original class with the differential that now it can receive as input the entities’ positions and avoid removing them when creating the samples. The new version can be imported from extended_lime_explainer.py as EXtendedLimeTextExplainer().

Additionally, the work evaluates the performance of LIME in RE tasks by evaluating Qualitative and Quantitative metrics referenced from the literature that also required modification since they included the process of removing/replacing words from a sentence. Therefore it was ensured that the entities remain and their positions are also updated accordingly. 

All are evaluated, for comparison, in a complex neural model, called AGGCN, and a traditional machine learning model: Naive Bayes.

For clarification, in this document, the term "entities" refers to the pair of nominals in which the model intends to classify their relation. "Lime rationals"/"Lime features" is the main output of a LIME text-explanation, it consists of a list of words that represent the higher importance for the output.

## Dataset

The dataset chosen for this project is SemEval 2010 - task 8 (https://huggingface.co/datasets/sem_eval_2010_task_8). It comprises semantic relations between two nominals. There are 10.717 annotated samples and 9 classes ('Entity-Destination', 'Cause-Effect', 'Member-Collection', 'Entity-Origin', 'Message-Topic', 'Component-Whole', 'Instrument-Agency', 'Product-Producer', 'Content-Container') plus the class 'Other'.



## AGGCN 

For the evaluation of the Neural model, AGGCN was chosen. The acronym stands for Attention Guided Graph Convolutional Networks for Relational Extraction task.

Model Repository: https://github.com/Cartus/AGGCN/tree/master/semeval 

### Requirements
python 3.9.6

tensorflow 2.12.0

torch 2.0.1

No need to re-train the model, please only follow the steps from the section "Preparation".

### Preparation 
Please follow these steps if the  GloVe vectors and Vocabulary are not present in the folder `/dataset` folder. Instructions extracted from https://github.com/Cartus/AGGCN/tree/master/semeval :

#### Dataset
Datasets in JSON format are already put under the directory  `Data/`.

#### GloVe vectors and vocab
If GloVe vectors are not present in the folder `dataset/glove/` nor vocab in the folder `dataset/vocab/`, please follow the steps:

First, download and unzip GloVe vectors:

```
cd AGGCN
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:

(Be sure you are inside the AGGCN directory)

```

python3 prepare_vocab.py dataset/semeval dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

#### Universal Sentence Encoder
If there is no directory tf_hub_model or it is empty, please run the following Python code

```python
import tensorflow_hub as hub
import tensorflow as tf

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
print ("module %s loaded" % module_url)

tf.saved_model.save(model, "../tf_hub_model/") """
```

## Naive Bayes

The Naive Bayes model was implemented using scikit-learn library and followed the same configuration from the session **Requirements** from the AGGCN model.

## Metrics

The implemented metrics are demonstrated in the Jupyter notebooks under each model directory. 

For AGGCN: AGGCN/lime_evaluation_AGGCN.ipynb

For Naive Bayes: Naive_Bayes/lime_evaluation_NB.ipynb

Metrics implemented: 
* Faithfulness
  * Suffiency
    * Suffiency base
    * Suffiency AOPC
    * Suffiency AOPC Random
  * Comprehensiveness
    * Comprehensiveness base
    * Comprehensiveness AOPC
    * Comprehensiveness AOPC Random
* Stability:
  * Inherent Stability
  * Parameter Stability
* Intuitiveness

### Faithfulness
Faithfulness consists of evaluating 2 metrics: Comprehensiveness and Sufficiency. 
the metric is presented by DeYoung et. al. (2020) in the paper about the ERASER method (ERASER stands for Evaluating Rationales And Simple English Reasoning). 

Faithfulness is intended to assess how well the rationales provided by a model informed its predictions

* AOPC

Since the number of rationals is a parameter to be set in LIME, it is also calculated the metrics AOPC (“Area Over the Perturbation Curve”), a methodology also presented in the ERASER paper. Instead of choosing a determined value for the number of rationals, we repeat the process for several values, derived from percentages of the total number of words, and do the average of the outputs. 

* Random AOPC

To provide more insightful information, the calculated metrics are also compared with metrics that choose a random set of words as the modified sentence, instead of removing or keeping the rationals. 

#### Suffiency

Sufficiency consists of evaluating the output if only the rationals were present. In this way, we compare the explanation of the whole sentence with the one containing only the rationals and compare the prediction probabilities.

If the prediction probability is higher with only the rationals, it means they were sufficient for the prediction.

Sufficiency is therefore evaluated as the difference between the prediction probability of the whole sentence and the sentence with only the rationals. If the result is positive, it leads to the assumption that the rationales were not sufficient.

#### Comprehensiveness

Comprehensiveness complements the idea of sufficiency by instead of keeping only the rationals, it removes the rationals and keeps the rest of the words. It is intended to assess the degree to which the rationales selected all the required features to come to a prediction. 

### Stability
It followed the methodology proposed by “Are Your Explanations Reliable?" Investigating the Stability of LIME in Explaining Textual Classification Models via Adversarial Perturbation from C Burger, L Chen, T Le (2023).
#### Inherent Stability

For a certain sentence, it is tested the difference in explanation for sampling rates varying from 1000 to 10000, using 1000 as the step (default is 5000), then repeat the process for another random seed. Results are compared using RBO (Ranking-biased Overlap).
#### Parameter Stability
Consists of perturbing a sentence by changing unimportant words with their synonym and evaluating the explanations in comparison with the original sentence.

Steps: 

* Define the Rationals and the Relation entities - as they will not change.
* From the left words, change the words by their synonym, using paragram-ws353. Ensure that words changed are of the same Part of the Speech (POS). Additionally, it is allowed to change verbs with nouns (and vice-versa). 
* Calculate the similarity between the original and the perturbed sentence 
* Repeat the process until the similarity is lowest possible but above 0.5, or after 5 iterations. Outputs the perturbed sentence.
* Calculate the explanation for the original and perturbed sentence, and compare the results using RBO. If RBO < 0.5, the "attack" was successful, meaning that the perturbed sentence performed significantly differently from the original one. 
* With the function rate_succeful_attackss, repeat the above steps for several sentences and get the percentage of successful attacks.
### Intuitiveness 
It is a qualitative metric that analyzes whether the selected features by LIME are reasonable with reality. It means that the features should agree with human intuition about the class output. In other words, is evaluated to which degree the rationals agree with the relation class.







