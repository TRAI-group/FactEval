'''
Model cards
meta-llama/Meta-Llama-3-8B for Llama
google/gemma-1.1-7b-it for Gemma
mistralai/Mistral-7B-Instruct-v0.3 for Mistral
'''

from transformers import AutoModelForCausalLM, AutoTokenizer,GPTJModel
import os, json, random
import numpy as np

from huggingface_hub import login

# login(token=" ",add_to_git_credential=True)
# huggingface-cli cache clear


import csv
import torch


# Open the file in write mode
with open("test_predictions_gemma_cot.csv", "w", newline='') as f:
    # Create a CSV writer object
    writer = csv.writer(f)
    # Write the header row
    writer.writerow(["id", "claim", "evidence", "actual_label", "predicted_label","prompt"])
pass

# Load dataset from JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


test_data = load_jsonl('data/fever_test.jsonl')

def create_prompt(claim, evidence):
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


model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it", token='hf_token',cache_dir="path_to_directory").cuda()
# model.to(device)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it",token='hf_token',cache_dir="path_to_directory")

# Load test data
test_samples = test_data  # Adjust according to your dataset

# Evaluate predictions
true_labels = [d['label'] for d in test_samples]
predictions = []

model.generation_config.pad_token_id = tokenizer.pad_token_id

import re
cnt=0
def extract_first_claim_class(generated_text,prompt):

	# Split the decoded_output by "Answer:"
	answers = generated_text.split("Answer:")#"Classify the above Claim into one of the three classes: SUPPORTED, REFUTED or NEI based on the above Evidence:")
	
	predicted_class = answers[-1]  
	if(("NEI" in predicted_class) | ("nei" in predicted_class) | ("not enough information" in predicted_class)):
		return "NEI"
	elif(("SUPPORTED" in predicted_class) | ("supported" in predicted_class) | ("supports" in predicted_class) | ("does not support" in predicted_class)):
		return "SUPPORTED"
	elif(("REFUTED" in predicted_class) | ("refuted" in predicted_class) | ("refutes" in predicted_class)):
		return "REFUTED"
	elif("cannot be supported" in predicted_class):
		return "NEI"
	elif(("does not provide sufficient information" in predicted_class) | ("does not provide sufficient information" in predicted_class)):
		return "NEI"
	else:
		# print("Errrrrrrrrrorrrr")
		return predicted_class

	
i=0
for example in test_samples:
	i=i+1
	
	prompt = create_prompt(example['claim'], example['gold_evidence_text'])
	
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

	
	generated_ids = model.generate(input_ids, do_sample=True, temperature=0.3, max_length=input_ids.shape[1] + 150)#max_new_tokens=5)#max_length=input_ids.shape[1] + 6)#,skip_special_tokens=True)
	
	generated_text = tokenizer.decode(generated_ids[0])
	claim_class = extract_first_claim_class(generated_text, prompt)
	
	# if(claim_class not in ["SUPPORTED", "REFUTED","refutes","NEI","does not support","supports","not enough information","nei","supported","refuted","cannot be supported","does not provide sufficient information","does not provide sufficient information"]):
	# 	cnt=cnt+1
	# 	print("**************counter with None class*********", cnt)
	# 	print(claim_class)

	print("class",claim_class)

	with open("test_predictions_gemma_cot.csv", "a", newline='') as f:
    	# Create a CSV writer object
	    writer = csv.writer(f)
	    
	    # Write the header row
	    writer.writerow([example['id'], example['claim'], example['gold_evidence_text'], example['label'], claim_class, generated_text])
	pass
	