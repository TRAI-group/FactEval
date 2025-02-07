import json
import numpy as np
from tqdm import tqdm
import datasets
import shap,torch,csv
import scipy as sp
from datasets.dataset_dict import DatasetDict
from sklearn import metrics
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np

metric = evaluate.load("accuracy")
seed = 98

import itertools
import random
from typing import List


from perturbations_in_the_wild_main.anthro_lib import ANTHRO
import random
anthro = ANTHRO()
anthro.load('perturbations_in_the_wild_main/ANTHRO_Data_V1.0')


class PhoneticPerturbations():
    
    def __init__(
        self, seed: int = 0, max_outputs: int = 1, perturb_pct: float = 0.50
    ) -> None:
        
        
        self.perturb_pct = perturb_pct
        self.seed=seed
        self.max_outputs=max_outputs
    # def pho_50(word):
    #     perturbed_texts = []
    #     aa=anthro.get_similars(word, level=1, distance=5, strict=True)
    #     # print("lissssssss",word, aa)
    #     ch=[]
    #     for a in aa:
    #         # print(a.lower())

    #         if(a.lower()!=word.lower()):
    #             # print()
    #             ch.append(a.lower())
        
    #     if(len(ch)==0):
    #         k=word
    #         # return k
    #     else:
    #         k=random.choice(ch)
    #         print("per***********orig nd perturb",word,k, ch)
        

    #     return k
    
    def pho_50(self, sentence: str) -> List[str]:
        random.seed(self.seed)
        sentence_list = sentence.split()#list(sentence)
        l=len(sentence_list)
        k_fifty=int(l/2)
        k_twntyfive=int(l/4)
        if(k_twntyfive==0):
            k_twntyfive=1
        if(k_fifty==0):
            k_fifty=1
        # print("sen list", sentence_list, l,k_fifty)
        perturbed_texts = []
        
        cnt=1
        for idx, word in enumerate(sentence_list):
            if(cnt>k_fifty):
                break 
            else:
                # try:
                word=word.strip()
                aa=anthro.get_similars(word, level=1, distance=5, strict=True)
                ch=[]
                # ch=[]
                for a in aa:
                    # print(a.lower())

                    if(a.lower()!=word.lower()):
                        # print()
                        ch.append(a.lower())
                # print("listss",ch,word)
                if(len(ch)==0):
                    k=word
                    # return k
                else:
                    k=random.choice(ch)
                    print("per***********orig nd perturb",word,k,ch)        
                sentence_list[idx]=k
                
                cnt=cnt+1
                

        perturbed_texts=" ".join(sentence_list)

        return perturbed_texts

    def pho_25(self, sentence: str) -> List[str]:
        random.seed(self.seed)
        sentence_list = sentence.split()#list(sentence)
        l=len(sentence_list)
        k_fifty=int(l/2)
        k_twntyfive=int(l/4)
        if(k_twntyfive==0):
            k_twntyfive=1
        if(k_fifty==0):
            k_fifty=1
        # print("sen list", sentence_list, l,k_fifty)
        perturbed_texts = []
        
        cnt=1
        for idx, word in enumerate(sentence_list):
            if(cnt>k_twntyfive):
                break 
            else:
                # try:
                word=word.strip()
                aa=anthro.get_similars(word, level=1, distance=5, strict=True)
                ch=[]
                # ch=[]
                for a in aa:
                    # print(a.lower())

                    if(a.lower()!=word.lower()):
                        # print()
                        ch.append(a.lower())
                # print("listss",ch,word)
                if(len(ch)==0):
                    k=word
                    # return k
                else:
                    k=random.choice(ch)
                    print("per***********orig nd perturbed",word,k,ch)        
                sentence_list[idx]=k
                
                cnt=cnt+1
                

        perturbed_texts=" ".join(sentence_list)

        return perturbed_texts

    

def read_data(path, to_dataset=True):
    '''
    read csv and return list of instances
    '''
    rows = []; label_dict = {'SUPPORTED': 0, 'REFUTED': 1, 'NEI': 2}
    # read data
    with open(path, 'r') as f:
        j_list = list(f)

    for j_str in j_list:
        row_dict = json.loads(j_str)
        
        rows.append({
            'id': row_dict['id'],
            'text': row_dict['gold_evidence_text'].strip() + '[SEP]' + row_dict['claim'].strip(),
            'label': label_dict[row_dict['label']],
            'claim': row_dict['claim'].strip(),
            'evidence':  row_dict['gold_evidence_text'].strip()

        })
    if to_dataset:
        # to object
        rows = datasets.Dataset.from_list(rows)

    return rows



      
if __name__ == "__main__":
    # dataset
    rows = read_data("./fever_test.jsonl", to_dataset=False)


    functionss2=['pho_25','pho_50']

    ll = PhoneticPerturbations()
   
   
    prev_pred = []; gold = []
    for func in functionss2:
        aa=func
        with open("data/adv_data/"+aa+"_adv_test.csv", "w", newline='') as f:
            # Create a CSV writer object
            writer = csv.writer(f)	    
            # Write the header row
            writer.writerow(["id", "actual_claim", "gold_evidence_text", "adv_claim", "label"])
        pass

    stop=0
    for row in tqdm(rows, total=len(rows)):
        # if(stop==50):
        #     break
        stop=stop+1

        print("row", row['claim'])
        for func in functionss2:
            aa=func
            c=ll
            
            if(aa=="pho_25"):
                #perturb claim
                aa=func
                # print("function",aa)
                func = getattr(c, func, None)
                # print("func",func)
                pert=1; budget1 = 0.25; perturb_ratio=0
                act=row['claim']
                
                ph_claim=func(act) 
                
                
                with open("data/adv_data/"+aa+"_adv_test.csv", "a", newline='') as f:
                    # Create a CSV writer object
                    writer = csv.writer(f) 
                    # Write the header row
                    writer.writerow([row['id'], act, row['evidence'], ph_claim, row['label']])
                pass

            elif(aa=="pho_50"):
                act=row['claim']
                
                #perturb claim
                aa=func
                # print("function",aa)
                func = getattr(c, func, None)
                
                pert=1; budget1 = 0.50; perturb_ratio=0
                # homo_claim=ll.homo_50(act) 
                ph_claim=func(act) 
                
                with open("data/adv_data/"+aa+"_adv_test.csv", "a", newline='') as f:
                    # Create a CSV writer object
                    writer = csv.writer(f) 
                    # Write the header row
                    writer.writerow([row['id'], act, row['evidence'], ph_claim, row['label']])
                pass

       
