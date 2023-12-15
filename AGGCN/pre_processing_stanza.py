import stanza
import json


stanza_nlp = stanza.Pipeline(lang='en', be_quiet=True)


def pre_processing(texts, subj_entity_word, obj_entity_word, stanza_nlp=stanza_nlp, relation= "Message-Topic"):
    #get original list

    pre_processed_list = []
    j = 0
    n = 0

    for text in texts:
        
        id = j
       
        words = text.split()
        if len(words) < 2:
            continue 

        #get entities postion
        subj_start = words.index(subj_entity_word)
        obj_start = words.index(obj_entity_word)
       
        doc = stanza_nlp(text)
       
        for i, sent in enumerate(doc.sentences):
            head =[]
            pos=[]
            tokens= []
            deprel = []
            for word in sent.words:
                head.append(word.head)
                pos.append(word.xpos)
                tokens.append(word.text)
                deprel.append(word.deprel)



        #create dictionary
        pre_processed_sample = {"id": str(id),
                                "relation": relation, 
                                "token": tokens,
                                "stanford_pos": pos,
                                "stanford_head": head,
                                "stanford_deprel": deprel,
                                #"subj_start": subj_start[j],
                                #"subj_end": subj_start[j],
                                "subj_start": subj_start,
                                "subj_end": subj_start,
                                "obj_start": obj_start,
                                "obj_end": obj_start} 
                               # "obj_start": obj_start[j],
                                #"obj_end": obj_start[j]} 
        
        pre_processed_list.append(pre_processed_sample)
        j = j +1

        n = n +1
    
    #convert to json and write to file
    with open("dataset/semeval/pre_processed_data_.json", "w") as f:
        json.dump(pre_processed_list, f) 


    return pre_processed_list