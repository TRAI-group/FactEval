import os, json, random
import numpy as np
import csv
from using_checklist import FactTemplates
ft=FactTemplates()

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# train_data = load_jsonl('data/fever_train.jsonl')
test_samples = load_jsonl('fever_test.jsonl')

functionss=['contractions','expansions','jumble','synonym_adjective','subject_verb_dis','number2words', 'repeat_phrases']
for func in functionss:
	with open(func+"_test_set.csv", "w", newline='') as f:
    	# Create a CSV writer object
	    writer = csv.writer(f)
	    
	    # Write the header row
	    writer.writerow(["id", "actual_claim", "gold_evidence_text", "adv_claim", "label"])
	pass

thresh=[1,2]
for th in thresh:
	with open(str(th)+"_typos_test_set.csv", "w", newline='') as f:
    	# Create a CSV writer object
	    writer = csv.writer(f)
	    
	    # Write the header row
	    writer.writerow(["id", "actual_claim", "gold_evidence_text", "adv_claim", "label"])
	pass

for example in test_samples:

	for func in functionss:
		func1 = getattr(ft, func, None)
		adv_claim=func1(example['claim'])
		# folder="few_shot_attack_format/"+func
		with open(func+"_test_set.csv", "a", newline='') as f:
	    	# Create a CSV writer object
		    writer = csv.writer(f)
		    
		    # Write the header row
		    writer.writerow([example['id'], example['claim'], example['gold_evidence_text'], adv_claim, example['label']])
		pass

	
	for th in thresh:
		adv_claim=ft.typos(example['claim'], th)
	
		with open(str(th)+"_typos_test_set.csv", "a", newline='') as f:
	    	# Create a CSV writer object
		    writer = csv.writer(f)
		    
		    # Write the header row
		    writer.writerow([example['id'], example['claim'], example['gold_evidence_text'], adv_claim, example['label']])
		pass
