from transformers import AutoModelForCausalLM, AutoTokenizer,GPTJModel
import os, json, random
import numpy as np
from huggingface_hub import login
import csv
import torch

# Open the file in write mode
with open("test_predictions_zero_shot_mistral.csv", "w", newline='') as f:
    # Create a CSV writer object
    writer = csv.writer(f)
    # Write the header row
    writer.writerow(["id", "claim", "evidence", "actual_label", "predicted_label","prompt"])
pass

# Load dataset from JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def create_prompt(claim, evidence):
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
    prompt += f"Claim: [{claim}]\nEvidence: [{evidence}]\nChoose one answer out of the three clasess: SUPPORTED, REFUTED or NEI. Answer:"
    return prompt

def extract_claim_class(generated_text):
	
	# Split the decoded_output by "Answer:"
	answers = generated_text.split("Answer:")

	predicted_class = answers[-1].split("\n")[0].strip() 

	print("Predicted Class:", predicted_class)

	return predicted_class


model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=' ',cache_dir=" ").cuda()
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",token=' ',cache_dir=" ")

# Load test data
test_samples = load_jsonl('data/fever_test.jsonl')
print("length of file", len(test_samples))

# Evaluate predictions
true_labels = [d['label'] for d in test_samples]
predictions = []

model.generation_config.pad_token_id = tokenizer.pad_token_id


i=0
for example in test_samples:
	i=i+1
	prompt = create_prompt(example['claim'], example['gold_evidence_text'])
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

	generated_ids = model.generate(input_ids, do_sample=True, temperature=0.3, max_length=input_ids.shape[1] + 4)#max_new_tokens=5)#max_length=input_ids.shape[1] + 6)#,skip_special_tokens=True)
	
	generated_text = tokenizer.decode(generated_ids[0])

	claim_class = extract_claim_class(generated_text)

	with open("test_predictions_zero_shot_mistral.csv", "a", newline='') as f:
    	# Create a CSV writer object
	    writer = csv.writer(f)
	    
	    # Write the header row
	    writer.writerow([example['id'], example['claim'], example['gold_evidence_text'], example['label'], claim_class, generated_text])
	pass