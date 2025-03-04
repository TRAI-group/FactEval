import itertools
import random,torch
from typing import List
from perturbations_in_the_wild_main.anthro_lib import ANTHRO
import random
import itertools
import random
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer,GPTJModel
import os, json, random
import numpy as np
import csv


anthro = ANTHRO()
anthro.load('perturbations_in_the_wild_main/ANTHRO_Data_V1.0')


class PhoneticPerturbations():
    
    def __init__(
        self, seed: int = 0, max_outputs: int = 1, perturb_pct: float = 0.50
    ):
        """
        In order to generate multiple different perturbations, you should set seed=None
        """
        # super().__init__(seed=seed, max_outputs=max_outputs)
        self.perturb_pct = perturb_pct
        self.seed=seed
        self.max_outputs=max_outputs

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
       
        perturbed_texts = []
       
        cnt=1
        for idx, word in enumerate(sentence_list):
            if(cnt>k_fifty):
                
                break
                
            else:
                try:
                    aa=anthro.get_similars(word, level=1, distance=5, strict=True)
                    ch=[]
                    for a in aa:
                        
                        if(a.lower()!=word.lower()):
                            ch.append(a.lower())
                    k=random.choice(ch)
                    print("per***********",k)
                    sentence_list[idx]=k
                    cnt=cnt+1
                except:
                    continue

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
        
        perturbed_texts = []
        
        cnt=1
        for idx, word in enumerate(sentence_list):
            

            if(cnt>k_twntyfive):
                
                break
                
            else:
                print(word)
                try:
                    aa=anthro.get_similars(word, level=1, distance=5, strict=True)
                    ch=[]
                    for a in aa:
                        # print(a.lower())
                        if(a.lower()!=word.lower()):
                            ch.append(a.lower())
                    k=random.choice(ch)
                    print("per***********",k)
                    sentence_list[idx]=k
                    cnt=cnt+1
                except:
                    continue
                
                

        perturbed_texts=" ".join(sentence_list)

        return perturbed_texts



# Load dataset from JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]



def create_prompt_zero(claim, evidence):
    
    prompt = """
            <start_of_turn>user
            The claim is given in the form Claim: [claim], Evidence: [evidence]. You are an intelligent fact checker trained on Wikipedia. Your task is to verify the claim based on the given evidence. There are three available classes.\n
            {
            "SUPPORTED": If the evidence supports the claim.\n
            "REFUTED": If the evidence contradicts the claim.\n
            "NEI": If the evidence does not provide sufficient information to determine the claim's validity.\n
            }
            ### Important:
            - **Only choose one class from above mentioned classes.\n
            """

    
    prompt +=f"\nClaim: [{claim}]\nEvidence: [{evidence}]\nChoose one answer out of the three clasess: SUPPORTED, REFUTED or NEI<end_of_turn>\n<start_of_turn>model\nAnswer: "
    return prompt
    
def create_prompt_cot(claim, evidence):
    prompt = "";current="";print("********************")
    prompt = """
            The claim is given in the form Claim: [claim], Evidence: [evidence]. You are an intelligent fact checker trained on Wikipedia. Your task is to verify the claim based on the given evidence. There are three available classes.\n
            {
            "SUPPORTED": If the evidence supports the claim.\n
            "REFUTED": If the evidence contradicts the claim.\n
            "NEI": If the evidence does not provide sufficient information to determine the claim's validity.\n
            }
            ### Important:
            - **Only choose one class from above mentioned classes.\n
            """

        
    prompt += f"Claim: [{claim}]\nEvidence: [{evidence}]\nWhat should be the final class from SUPPORTED, REFUTED or NEI? Answer: Lets think step by step."
    
    return prompt


def read_prompt_file_fewshot(claim, evidence): #few-shot

    prompt = ""
    with open('prompt.json', 'r') as file:
        prompt = json.load(file)
    prompt += f"<start_of_turn>user\nClaim: [{claim}]\nEvidence: [{evidence}]<end_of_turn>\n<start_of_turn>model\nAnswer: "
  

    return prompt

model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it", token=' ',cache_dir=" ").cuda()
# model.to(device)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it",token=' ',cache_dir=" ")

# Load test data
test_samples = load_jsonl('data/fever_test.jsonl')


# Evaluate predictions
true_labels = [d['label'] for d in test_samples]
predictions = []

model.generation_config.pad_token_id = tokenizer.pad_token_id
import re
def extract_first_claim_class(generated_text,prompt):
   

    # Split the decoded_output by "Answer:"
    answers = generated_text.split("Answer:")

   
    predicted_class = answers[-1]


    print("Predicted Class:", predicted_class)

    return predicted_class


functionss2=['pho_25','pho_50']

ll=PhoneticPerturbations()


i=0
for example in test_samples:
    i=i+1 #47
   
    print(i)
    for func in functionss2:
        
        aa=func
        func = getattr(ll, func, None)

        
        claim = func(example['claim'])
       
        prompt = create_prompt_zero(claim, example['gold_evidence_text'])
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda() #.to(device)

        
        if(len(input_ids[0])>8192):
            
            truncated_input_ids = input_ids[:, :8180]
            
            generated_ids = model.generate(truncated_input_ids, do_sample=True, temperature=0.2, max_length=input_ids.shape[1] + 6)
        else:
            generated_ids = model.generate(input_ids, do_sample=True, temperature=0.2, max_length=input_ids.shape[1] + 6)


        generated_text = tokenizer.decode(generated_ids[0])

        claim_class = extract_first_claim_class(generated_text,prompt)

        # print(claim_class)
        folder="attack_gemma_phonetic/"+aa
        with open(folder+"_gemma_json.csv", "a", newline='') as f:
            # Create a CSV writer object
            writer = csv.writer(f)
            
            # Write the header row
            writer.writerow([example['id'], example['claim'], claim, example['gold_evidence_text'], example['label'], claim_class,generated_text])
        pass

        

        


