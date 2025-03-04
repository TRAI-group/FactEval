from transformers import AutoModelForCausalLM, AutoTokenizer,GPTJModel
import os, json, random
import numpy as np
import csv
import torch
from using_checklist import FactTemplates
from using_checklist import StressTest, NLPPerturbation
import re

# Load dataset from JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]



#perturbations
ft=StressTest()
ft1=NLPPerturbation()

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


tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it",token=' ',cache_dir=" ")

# Load test data
test_samples = load_jsonl('data/fever_test.jsonl')  
print("length of file", len(test_samples))


# Evaluate predictions
true_labels = [d['label'] for d in test_samples]
predictions = []

model.generation_config.pad_token_id = tokenizer.pad_token_id

def extract_claim_class(generated_text,prompt):
	
	# Split the decoded_output by "Answer:"
	answers = generated_text.split("Answer:")
	
	predicted_class = answers[-1].split("\n")[0].strip()  
	
	print("Predicted Class:", predicted_class)

	return predicted_class

#StressTest perturbations
i=0
functionss1=['perturb_swap','addition']
for example in test_samples:
	i=i+1 
	
	print(i)
	for func in functionss1:
		#perturb claim
		aa=func
		func = getattr(ft, func, None)

		
		if(aa=="perturb_swap"):
			claim=func(example['claim'],1)
		else:
			claim= func(example['claim'])
		

		prompt = create_prompt(claim, example['gold_evidence_text'])
		
		input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda() 
		
		generated_ids = model.generate(input_ids, do_sample=True, temperature=0.2, max_length=input_ids.shape[1] + 6)

		generated_text = tokenizer.decode(generated_ids[0])

		claim_class = extract_claim_class(generated_text,prompt)

		
		folder="attack_gemma_stressNLP/"+aa
		with open(folder+"_gemma_stress_NLPper.csv", "a", newline='') as f:
	    	# Create a CSV writer object
		    writer = csv.writer(f)
		    
		    # Write the header row
		    writer.writerow([example['id'], example['claim'], claim, example['gold_evidence_text'], example['label'], claim_class,generated_text])
		pass


		

functionss2=['char_delete','char_insert','char_rep','word_rep']

i=0
for example in test_samples:
	i=i+1 
	
	for func in functionss2:
		#perturb claim
		aa=func
		func = getattr(ft1, func, None)

		
		claim= func(example['claim'])
		
		prompt = create_prompt_zero(claim, example['gold_evidence_text'])
		
		input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda() #.to(device)

		
		generated_ids = model.generate(input_ids, do_sample=True, temperature=0.2, max_length=input_ids.shape[1] + 6)#,skip_special_tokens=True)


		generated_text = tokenizer.decode(generated_ids[0])

		claim_class = extract_first_claim_class(generated_text,prompt)

		print(claim_class)
		folder="attack_gemma_stressNLP/"+aa
		with open(folder+"_gemma_stress_NLPper.csv", "a", newline='') as f:
	    	# Create a CSV writer object
		    writer = csv.writer(f)
		    
		    # Write the header row
		    writer.writerow([example['id'], example['claim'], claim, example['gold_evidence_text'], example['label'], claim_class,generated_text])
		pass


		

		


