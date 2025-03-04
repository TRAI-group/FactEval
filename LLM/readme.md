# FactEval

---

## Attacks Overview  

This repository includes multiple attack modules implemented in Python, each introducing different types of text perturbations. Below is a detailed breakdown of the available attack functions categorized by their respective files:  

### **1. `attack_using_others.py`**  
This module contains various textual perturbations, including:  
- **`contractions`** – Expands or contracts words (e.g., "do not" → "don't").  
- **`expansions`** – Reverses contractions (e.g., "can't" → "cannot").  
- **`typos`** – Introduces common typographical errors.  
- **`jumble`** – Jumbles the order of characters within words.  
- **`synonym_adjective`** – Replaces adjectives with their synonyms.  
- **`subject_verb_dis`** – Introduces subject-verb disagreement errors.  
- **`number2words`** – Converts numbers to words (e.g., "10" → "ten").  
- **`repeat_phrases`** – Repeats phrases to introduce redundancy.  

### **2. `attack_using_phonetics.py`**  
This module applies **phonetic perturbations**. 

### **3. `attack_using_homo_leet.py`**  
This module includes:  
- **Homoglyph-based attacks** – Replaces characters with visually similar Unicode characters.  
- **LEET-based transformations** – Converts text to leetspeak.  

### **4. `attack_using_StressNLP.py`**  
This module includes character- and word-level perturbations:  

- **Function Set 1:**  
  - `perturb_swap` – Swaps adjacent characters.  
  - `addition` – Randomly adds characters.  

- **Function Set 2:**  
  - `char_delete` – Deletes random characters.  
  - `char_insert` – Inserts extra characters within words.  
  - `char_rep` – Repeats or replaces characters.  
  - `word_rep` – Replaces words with similar alternatives.  

---

