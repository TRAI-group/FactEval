from transformers import AutoModelForCausalLM, AutoTokenizer,GPTJModel
import os, json, random
import numpy as np
import torch
import csv
from using_checklist import FactTemplates



ft=FactTemplates()

# Load dataset from JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

print("length of file", len(test_data))
def create_prompt_zero(claim, evidence): #zero-shot
    
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
    
def create_prompt_cot(claim, evidence): #COT
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


model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it", token='',cache_dir="").cuda()#, force_download=True)#EleutherAI/gpt-j-6B").cuda()


tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it",token='',cache_dir="")#EleutherAI/gpt-j-6B")

# Load test data
test_samples = load_jsonl('data/fever_test.jsonl')

# Evaluate predictions
true_labels = [d['label'] for d in test_samples]
predictions = []

model.generation_config.pad_token_id = tokenizer.pad_token_id

def extract_claim_class(generated_text,prompt):
  
    # Split the decoded_output by "Answer:"
    answers = generated_text.split("Answer:")

    
    predicted_class = answers[-1]
    
    print("Predicted Class:", predicted_class)

    return predicted_class


i=0
functionss=['contractions','expansions','typos','jumble','synonym_adjective','subject_verb_dis','number2words', 'repeat_phrases']
for example in test_samples:
	i=i+1 
	
	print(i)
	for func in functionss:
		#perturb claim
		func1 = getattr(ft, func, None)

		claim= func1(example['claim'])
		

		prompt = create_prompt_zero(claim, example['gold_evidence_text'])
		
		input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda() #.to(device)
		
		generated_ids = model.generate(input_ids, do_sample=True, temperature=0.2, max_length=input_ids.shape[1] + 6)

		generated_text = tokenizer.decode(generated_ids[0])

		claim_class = extract_claim_class(generated_text,prompt)

		print(claim_class)
		folder="attack_gemma_checklist/"+func
		with open(folder+"_gemma_json.csv", "a", newline='') as f:
	    	# Create a CSV writer object
		    writer = csv.writer(f)
		    
		    # Write the header row
		    writer.writerow([example['id'], example['claim'], claim, example['gold_evidence_text'], example['label'], claim_class, generated_text])
		pass


		

		

		


