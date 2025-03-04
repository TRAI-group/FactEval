import os, json, random
import numpy as np
import itertools
import random
from typing import List
from using_checklist import FactTemplates
from using_checklist import StressTest, NLPPerturbation


# from anthro_lib import ANTHRO
# anthro = ANTHRO()
# file_path = os.path.join(pp, 'ANTHRO_Data_V1.0')
# anthro.load(file_path)




# Load dataset from JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# Sample a few instances for each class
def sample_data(data, num_samples_per_class):
    supported = [d for d in data if d['label'] == 'SUPPORTED']
    refuted = [d for d in data if d['label'] == 'REFUTED']
    nei = [d for d in data if d['label'] == 'NEI']
    
    print(len(supported), len(refuted), len(nei))
    samples = {
        'SUPPORTED': random.sample(supported, num_samples_per_class),
        'REFUTED': random.sample(refuted, num_samples_per_class),
        'NEI': random.sample(nei, num_samples_per_class)
    }

    return samples


test_data = load_jsonl('data/fever_test.jsonl')



def create_few_shot_prompt(few_shot_samples, funnc, name):
    prompt = ""
    prompt = """
            The claim is given in the form Claim: [claim], Evidence: [evidence], Answer: [Answer]. You need to given an answer in the [Answer] slot. There are three available answers that you can choose to fill the slot: SUPPORTED, REFUTED, NOT_ENOUGH_INFO.\n
            Your task is to classify the claim based on the evidence. Choose **only** one of the following labels:\n
            {
            SUPPORTED: If the evidence supports the claim.\n
            REFUTED: If the evidence contradicts the claim.\n
            NEI: If the evidence does not provide sufficient information to determine the claim's validity.\n
            }

           ### Important:
            - **Only choose one class from above mentioned classes. 
        
            ### Do not include any additional text or explanations.


            Here are some examples:\n"""

    for label, examples in few_shot_samples.items():
        for example in examples:
                if(name=="perturb_swap"):
                    exampleh=funnc(example['claim'],1)
                else:
                    exampleh= funnc(example['claim'])

                
                print("actual text", example['claim'])
                print("perturbed_texts",exampleh)
                prompt += f"""Claim: [{example['claim']}]\nEvidence: [{example['gold_evidence_text']}]\nAnswer: [{example['label']}]\n"""
                prompt += f"""Claim: [{exampleh}]\nEvidence: [{example['gold_evidence_text']}]\nAnswer: [{example['label']}]\n"""
   
    return prompt


ft=StressTest()
ft1=NLPPerturbation()

functionss2=['char_delete','char_insert','char_rep','word_rep']
train_samples = sample_data(train_data, 2)
for func in functionss2:
    funcc = getattr(ft1, func, None)
    prompt = create_few_shot_prompt(train_samples, funcc, func)
    folder= "adv_prompts/"
    with open(folder+func+"_prompt.json",'w') as file:
        json.dump(prompt, file, indent=4)


functionss1=['perturb_swap','addition']
for func1 in functionss1:
    funcc1 = getattr(ft, func1, None)
    
    prompt = create_few_shot_prompt(train_samples, funcc1, func1)
    folder= "adv_prompts/"
    with open(folder+func1+"_prompt.json", 'w') as file:
        json.dump(prompt, file, indent=4)

