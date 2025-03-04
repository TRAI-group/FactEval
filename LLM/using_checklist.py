
from checklist.perturb import Perturb
import spacy
spacy.__version__

from num2words import num2words
import re
import spacy
from checklist.editor import Editor
from checklist.perturb import Perturb
import random
from num2words import num2words
import nltk
import numpy as np
from random import seed
from random import randint
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')




class BaseTemplate:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = self.nlp.Defaults.stop_words

    

    def typos(self, sent):
        return Perturb.add_typos(sent)

    def contractions(self, sent):
        x = Perturb.contract(sent)
        return x if x != sent else sent

    def expansions(self, sent):
        x = Perturb.expand_contractions(sent)
        return x if x != sent else sent

    


    def jumble(self, sent):
        tokens = [i.text for i in self.nlp(sent)]
        random.shuffle(tokens)
        return ' '.join(tokens)

    
    
    def synonym_adjective(self, sent):
        pos = nltk.pos_tag(nltk.word_tokenize(sent))
        flag = 0
        sen =[]
        for i in range(len(pos)):
            w, p = pos[i]
            print("w nd p", w,p)
            if p in ['JJ', 'JJR', 'JJS']:
                try:
                    print("try")
                    syn = Editor().synonyms(sent, w)
                except:
                    print("else")
                    syn = []
                if len(syn) > 0:
                    sen.append(syn[0])
                    flag = 1
                else:
                    sen.append(w)
            else:
                sen.append(w)
        if flag == 1:
            out = " ".join(x for x in sen)
            return out
        return sent

    

    def subject_verb_dis(self, sent):
        cases = {'was':'were', 
                'were':'was', 
                'is':'are',
                'are':'is', 
                'has':'have',
                'have':'has',
                'does':'do',
                'do':'does'}
        sentence =''
        doc = self.nlp(sent)
        for i in doc:
            if i.pos_ =="AUX":
                try:
                    w = cases[i.text]
                except:
                    w =i.text
                sentence  = sentence + w + ' '
            else:
                sentence = sentence + i.text + ' '
        return sentence.strip()
        
    def number2words(self, sent):
        out = ''
        for i in sent.split(' '):
            if i.isdigit():
                out = out + num2words(i) + ' '
            else:
                out = out + i + ' '
        return out.strip()
    
    def repeat_phrases(self, sent):
        pos = nltk.pos_tag(nltk.word_tokenize(sent))
        sen = []
        l = len(pos)
        rep_word = ''
        flag = 0
        for i in range(l-1):
            w, p = pos[i]
            if i< l*0.25:
                rep_word += " " + w
                flag = 1
                sen.append(w)
            else:
                sen.append(w)
        sen.append(pos[l-1][0])
        sen.append(rep_word)
        if flag==1: 
            out = " ".join(w for w in sen)
            return out
        return sent

    



class FactTemplates(BaseTemplate):
    def __init__(self):
        super(FactTemplates, self).__init__()

    def contractions(self, sent):
        return super().contractions(sent)
        
    def expansions(self, sent):
        return super().expansions(sent)
    
    def expansions(self, sent):
        return super().expansions(sent)

    

    def typos(self, sent):
        return super().typos(sent)

    
        

    def jumble(self, sent):
        return super().jumble(sent)

    

    def synonym_adjective(self, sent):
        return super().synonym_adjective(sent)

    

    def subject_verb_dis(self, sent):
        return super().subject_verb_dis(sent)

    def number2words(self, sent):
        return super().number2words(sent)

    
    
    def repeat_phrases(self, sent):
        return super().repeat_phrases(sent)

    @staticmethod
    def preprocess_sent(i):
        return re.sub(r'[FS]S *:', '', i).strip()

   

dg=FactTemplates()


class StressTest():
    def __init__(self):
        pass
   
    def perturb_swap(self, sent, threshold):
        '''
        swap characters within words
        '''
        count = 0
        
        def perturb_word_swap(word):
            if len(word) == 2:
                new_word = word[::-1]
            elif len(word) == 1:
                new_word = word
            else:
                char_ind = int(np.random.uniform(0, len(word) - 1))
                new_word = list(word)
                first_char = new_word[char_ind]
                new_word[char_ind] = new_word[char_ind + 1]
                new_word[char_ind + 1] = first_char
                new_word = "".join(new_word)
            return new_word
            
        sent=sent.lower()
        words = sent.split()
        for word in words: #threshold is number of words to be perturbed
            new_word = perturb_word_swap(word)
            if(count>threshold):
                break
            if(word!=new_word):
                print("uuuu",word, new_word, word !=new_word, sent.replace(word, new_word),count)
                
                sent = sent.replace(word, new_word)
                count=count+1
                # print("new_word",sent)
            
        return sent
        
    
    
    def perturb_sent_kb(self, sent, threshold):
    

    def addition(self, sent):
        """
        Adding right-sided padding.
        """
        addition = " and true is true. and true is true. and true is true." #!!!!!!
        sent = sent + addition
        return sent
            
'''
Stress Test Evaluation for Natural Language Inference
extract claims where we have numbers-> label preserving -> randomly choose one numeric entity from claim, change it and then add prefix less
than or greater than according to new number.
Label alter: do not change the number, just add less than or greater than to it.
'''



class NLPPerturbation():
    '''
    Evaluating the Robustness of Neural Language Models to Input Perturbation
    https://github.com/mmoradi-iut/NLP-perturbation
    
    '''
    def __init__(self):
        pass

    def return_random_number(self, begin, end):
            return randint(begin, end)
    
    def char_delete(self, sample_text):
        # sample_text = row[0]
        # sample_label = row[1]
        sample_tokenized = nltk.word_tokenize(sample_text)
    
        random_word_index = 0
        random_word_selected = False
    
        while (random_word_selected != True):
            random_word_index = self.return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
        #--------------------------- select a random position
            
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = self.return_random_number(1, len(selected_word)-2)
        print('Random position:', random_char_index)
        print('Character to delete:', selected_word[random_char_index])
        
        #--------------------------- delete the character
    
        temp_word = selected_word[:random_char_index]
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        print('After deletion:', perturbed_word)

        #--------------------------- reconstruct the perturbed sample
            
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        # is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        print('Perturbed sample:', perturbed_sample)
        
        # if (is_sample_perturbed == True):
        return perturbed_sample
        # else:
        #     return sample_text
            
    def char_insert(self, sample_text):    
        sample_tokenized = nltk.word_tokenize(sample_text)
    
        random_word_index = 0
        random_word_selected = False
    
        while (random_word_selected != True):
            random_word_index = self.return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = self.return_random_number(1, len(selected_word)-2)
        print('Random position:', random_char_index)
        
        #--------------------------- select a random character
        random_char_code = self.return_random_number(97, 122)
        print('Random character:', chr(random_char_code))
    
        temp_word = selected_word[:random_char_index]
        temp_word += chr(random_char_code)
        temp_word += selected_word[random_char_index:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        print('After insertion:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        print('Perturbed sample:', perturbed_sample)
        return perturbed_sample
        
    def char_rep(self, sample_text):
        is_sample_perturbed = False   
        sample_tokenized = nltk.word_tokenize(sample_text)
    
        random_word_index = 0
        random_word_selected = False
    
        while (random_word_selected != True):
            random_word_index = self.return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = self.return_random_number(1, len(selected_word)-2)
        print('Random position:', random_char_index)
        print('Character to repeat:', selected_word[random_char_index])
        
        #--------------------------- repeat the character
    
        temp_word = selected_word[:random_char_index]
        temp_word += selected_word[random_char_index] + selected_word[random_char_index]
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        print('After repetition:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        print('Perturbed sample:', perturbed_sample)
        
        if (is_sample_perturbed == True):
            return perturbed_sample
        else:
            sample_text

    
    def char_replacement(self, sample_text): #already in stress test
        
        def return_adjacent_char(input_char):
            if (input_char == 'a'):
                return 's'
            
            elif (input_char == 'b'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'v'
                else:
                    return 'n'
                
            elif (input_char == 'c'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'x'
                else:
                    return 'v'
                
            elif (input_char == 'd'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 's'
                else:
                    return 'f'
                
            elif (input_char == 'e'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'w'
                else:
                    return 'r'
                
            elif (input_char == 'f'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'd'
                else:
                    return 'g'
                
            elif (input_char == 'g'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'f'
                else:
                    return 'h'
                
            elif (input_char == 'h'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'g'
                else:
                    return 'j'
                
            elif (input_char == 'i'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'u'
                else:
                    return 'o'
                
            elif (input_char == 'j'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'h'
                else:
                    return 'k'
                
            elif (input_char == 'k'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'j'
                else:
                    return 'l'
            
            elif (input_char == 'l'):
                return 'k'
                
            elif (input_char == 'm'):
                return 'n'
                
            elif (input_char == 'n'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'b'
                else:
                    return 'm'
                
            elif (input_char == 'o'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'i'
                else:
                    return 'p'
                
            elif (input_char == 'p'):
                return 'o'
            
            elif (input_char == 'q'):
                return 'w'
                
            elif (input_char == 'r'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'e'
                else:
                    return 't'
                
            elif (input_char == 's'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'a'
                else:
                    return 'd'
                
            elif (input_char == 't'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'r'
                else:
                    return 'y'
                
            elif (input_char == 'u'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'y'
                else:
                    return 'i'
            
            elif (input_char == 'v'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'c'
                else:
                    return 'b'
                
            elif (input_char == 'w'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'q'
                else:
                    return 'e'
                
            elif (input_char == 'x'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'z'
                else:
                    return 'c'
                
            elif (input_char == 'y'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 't'
                else:
                    return 'u'
                
            elif (input_char == 'z'):
                return 'x'
            #---------------------------------------------
            elif (input_char == 'A'):
                return 'S'
            
            elif (input_char == 'B'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'V'
                else:
                    return 'N'
                
            elif (input_char == 'C'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'X'
                else:
                    return 'V'
                
            elif (input_char == 'D'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'S'
                else:
                    return 'F'
                
            elif (input_char == 'E'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'W'
                else:
                    return 'R'
                
            elif (input_char == 'F'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'D'
                else:
                    return 'G'
                
            elif (input_char == 'G'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'F'
                else:
                    return 'H'
                
            elif (input_char == 'H'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'G'
                else:
                    return 'J'
                
            elif (input_char == 'I'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'U'
                else:
                    return 'O'
                
            elif (input_char == 'J'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'H'
                else:
                    return 'K'
                
            elif (input_char == 'K'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'J'
                else:
                    return 'L'
            
            elif (input_char == 'L'):
                return 'K'
                
            elif (input_char == 'M'):
                return 'N'
                
            elif (input_char == 'N'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'B'
                else:
                    return 'M'
                
            elif (input_char == 'O'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'I'
                else:
                    return 'P'
                
            elif (input_char == 'P'):
                return 'O'
            
            elif (input_char == 'Q'):
                return 'W'
                
            elif (input_char == 'R'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'E'
                else:
                    return 'T'
                
            elif (input_char == 'S'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'A'
                else:
                    return 'D'
                
            elif (input_char == 'T'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'R'
                else:
                    return 'Y'
                
            elif (input_char == 'U'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'Y'
                else:
                    return 'I'
            
            elif (input_char == 'V'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'C'
                else:
                    return 'B'
                
            elif (input_char == 'W'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'Q'
                else:
                    return 'E'
                
            elif (input_char == 'X'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'Z'
                else:
                    return 'C'
                
            elif (input_char == 'Y'):
                which_adjacent = self.return_random_number(1, 2)
                if (which_adjacent == 1):
                    return 'T'
                else:
                    return 'U'
                
            elif (input_char == 'Z'):
                return 'X'
            
            else:
                return '*'

        is_sample_perturbed = False      

        sample_tokenized = nltk.word_tokenize(sample_text)
    
        random_word_index = 0
        random_word_selected = False
    
        while (random_word_selected != True):
            random_word_index = self.return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        char_is_letter = False
        tries_number = 0
        
        while (char_is_letter != True and tries_number <= 20):
            random_char_index = self.return_random_number(1, len(selected_word)-2)
            tries_number += 1
            if ((ord(selected_word[random_char_index]) >= 97 and ord(selected_word[random_char_index]) <= 122) or (ord(selected_word[random_char_index]) >= 65 and ord(selected_word[random_char_index]) <= 90)):
                char_is_letter = True
                is_sample_perturbed = True
        
        
        print('Random position:', random_char_index)
        print('Character to replace:', selected_word[random_char_index])
        
        #--------------------------- replace the character
    
        char_to_replace = selected_word[random_char_index]
        
        adjacent_char = return_adjacent_char(char_to_replace)
        
        print('Adjacent character:', adjacent_char)
        
        temp_word = selected_word[:random_char_index]
        temp_word += adjacent_char
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        print('After replacement:', perturbed_word)

        #--------------------------- reconstruct the perturbed sample    
        perturbed_sample = ""
        
        for i in range(0, random_word_index):     
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        print('Perturbed sample:', perturbed_sample,is_sample_perturbed)
        
        if (is_sample_perturbed == True):
            return perturbed_sample
        else:
            return sample_text

    def swap_chars(self, sample_text):
        def swap_characters(input_word, position, adjacent):
            temp_word = ''
            if (adjacent == 'left'):
                if (position == 1):
                    temp_word = input_word[1]
                    temp_word += input_word[0]
                    temp_word += input_word[2:]
                elif (position == len(input_word)-1):
                    temp_word = input_word[0:position-1]
                    temp_word += input_word[position]
                    temp_word += input_word[position-1]
                elif (position > 1 and position < len(input_word)-1):
                    temp_word = input_word[0:position-1]
                    temp_word += input_word[position]
                    temp_word += input_word[position-1]
                    temp_word += input_word[position+1:]
                
            elif (adjacent == 'right'):
                if (position == 0):
                    temp_word = input_word[1]
                    temp_word += input_word[0]
                    temp_word += input_word[2:]
                elif (position == len(input_word)-2):
                    temp_word = input_word[0:position]
                    temp_word += input_word[position+1]
                    temp_word += input_word[position]
                elif (position > 0 and position < len(input_word)-2):
                    temp_word = input_word[0:position]
                    temp_word += input_word[position+1]
                    temp_word += input_word[position]
                    temp_word += input_word[position+2:]
                
            return temp_word

        is_sample_perturbed = False
        
        sample_tokenized = nltk.word_tokenize(sample_text)
    
        random_word_index = 0
        random_word_selected = False
    
        while (random_word_selected != True):
            random_word_index = self.return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = self.return_random_number(0, len(selected_word)-1)
        print('Random position:', random_char_index)
        print('Char in random position:', selected_word[random_char_index])
        
        #--------------------------- select an adjacent for swapping
        adjacent_for_swapping = ''
        
        if (random_char_index == 0):
            adjacent_for_swapping = 'right'
        elif (random_char_index == len(selected_word)-1):
            adjacent_for_swapping = 'left'
        else:
            adjacent = return_random_number(1, 2)
            if(adjacent == 1):
                adjacent_for_swapping = 'left'
            else:
                adjacent_for_swapping = 'right'
                
        print('Adjacent for swapping:', adjacent_for_swapping)
        
        #--------------------------- swap the character and the adjacent
        temp_word = swap_characters(selected_word, random_char_index, adjacent_for_swapping)
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        print('After swapping:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        print('Perturbed sample:', perturbed_sample)
        
        if (is_sample_perturbed == True):
            return perturbed_sample
        else:
            return sample_text
    
        print('----------------------------------------------------------')

    
    def word_delete(self, sample_text):
        is_sample_perturbed = False
            
        sample_tokenized = nltk.word_tokenize(sample_text)
    
        random_word_index = 0
        random_word_selected = False
    
        while (random_word_selected != True):
            random_word_index = self.return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 1):
                random_word_selected = True
    
        print('Selected random word:', sample_tokenized[random_word_index])
        
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
            
        is_sample_perturbed = True
        
        print('Perturbed sample:', perturbed_sample)
        
        if (is_sample_perturbed == True):
            return perturbed_sample
        else:
            sample_text

    def word_rep(self, sample_text):
        is_sample_perturbed = False
        
        sample_tokenized = nltk.word_tokenize(sample_text)
    
        random_word_index = 0
        random_word_selected = False
    
        while (random_word_selected != True):
            random_word_index = self.return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 1):
                random_word_selected = True
    
        print('Selected random word:', sample_tokenized[random_word_index])

        selected_word = sample_tokenized[random_word_index]
        
        
        #--------------------------- reconstruct the perturbed sample
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += selected_word + ' ' + selected_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        print('Perturbed sample:', perturbed_sample)
        
        if (is_sample_perturbed == True):
            return perturbed_sample
        else:
            return sample_text

    def word_verb_tense(self, sample_text):
        
        def is_third_person(input_pos_tag):
            subject = ''
            for i in range(0, len(input_pos_tag)):
                token = input_pos_tag[i]
                if (subject == ''):
                    if (token[0].lower() in ('it', 'this', 'that', 'he', 'she')):
                        subject = 'third person'
                    elif (token[1] in ('NNP')):
                        subject = 'third person'
                    elif (token[0].lower() in ('i', 'we', 'you', 'they', 'she', 'these', 'those')):
                        subject = 'not third person'
                    elif (token[0].lower() in ('NNPS')):
                        subject = 'not third person'
            if (subject == 'third person'):
                return 'third person'
            elif (subject == 'not third person'):
                return 'not third person'
            else:
                return 'none'
        
        is_sample_perturbed = False  
        sample_tokenized = nltk.word_tokenize(sample_text)
        sample_pos_tag = nltk.pos_tag(sample_tokenized)
        
        print(sample_pos_tag)
        
        Perturbed_sample = ""
        
        remove_negation = False
        can_change_basic_form = True
        
        for i in range(0, len(sample_pos_tag)):
            token = sample_pos_tag[i]
            print(token[0], token[1])
            if (remove_negation == False and can_change_basic_form == True):
                
                if (token[0] == 'does' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, third person present simple
                    remove_negation = True
                    Perturbed_sample += "did not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'do' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, present simple
                    remove_negation = True
                    Perturbed_sample += "did not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'did' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, past simple
                    if (is_third_person(sample_pos_tag) == 'third person'):
                        remove_negation = True
                        can_change_basic_form = False
                        Perturbed_sample += "does not" + ' '
                        is_sample_perturbed = True
                    elif (is_third_person(sample_pos_tag) == 'not third person'):
                        remove_negation = True
                        Perturbed_sample += "do not" + ' '
                        is_sample_perturbed = True
                    
                elif (token[0] in ('is', 'am') and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += "was not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'are' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += "were not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'was' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    if (is_third_person(sample_pos_tag) == 'third person'):
                        remove_negation = True
                        Perturbed_sample += "is not" + ' '
                        is_sample_perturbed = True
                    elif (is_third_person(sample_pos_tag) == 'not third person'):
                        remove_negation = True
                        Perturbed_sample += "am not" + ' '
                        is_sample_perturbed = True
                        
                elif (token[0] == 'were' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += "are not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] in ('is', 'am')): #----- to be present and past, continuous present and past
                    Perturbed_sample += "was" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'are'): #----- to be present and past, continuous present and past
                    Perturbed_sample += "were" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'was'): #----- to be present and past, continuous present and past
                    if (is_third_person(sample_pos_tag) == 'third person'):
                        Perturbed_sample += "is" + ' '
                        is_sample_perturbed = True
                    elif (is_third_person(sample_pos_tag) == 'not third person'):
                        Perturbed_sample += "am" + ' '
                        is_sample_perturbed = True
                        
                elif (token[0] == 'were'): #----- to be present and past, continuous present and past
                    Perturbed_sample += "are" + ' '
                    is_sample_perturbed = True
            
                elif (token[1] == 'VBZ'): #----- third person singular present
                    verb = token[0]
                    length = len(verb)
                    if (verb == 'has'):
                        verb = 'have'
                    elif (verb[length-3:] == 'oes'):
                        verb = verb[:length-2]
                    elif (verb[length-4:] == 'ches'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'ses'):
                        verb = verb[:length-2]
                    elif (verb[length-4:] == 'shes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'xes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'zes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'ies'):
                        verb = verb[:length-3] + 'y'
                    else:
                        verb = verb[:length-1]
                        
                    past_tense = ""
                    
                    default_conjugator = mlconjug3.Conjugator(language='en')
                    past_verb = default_conjugator.conjugate(verb)
                    all_conjugates = past_verb.iterate()
                    
                    for j in range(0, len(all_conjugates)):
                        if (all_conjugates[j][1] == 'indicative past tense'):
                            past_tense = all_conjugates[j][3]
                    
                    Perturbed_sample += past_tense + ' '
                    is_sample_perturbed = True
                
                elif (token[1] == 'VBP'): #----- basic form present
                    verb = token[0]
                    
                    past_tense = ""
                    
                    default_conjugator = mlconjug3.Conjugator(language='en')
                    past_verb = default_conjugator.conjugate(verb)
                    all_conjugates = past_verb.iterate()
                    
                    for j in range(0, len(all_conjugates)):
                        if (all_conjugates[j][1] == 'indicative past tense'):
                            past_tense = all_conjugates[j][3]
                    
                    Perturbed_sample += past_tense + ' '
                    is_sample_perturbed = True
                
                elif (token[1] == 'VBD'): #----- past
                    if (is_third_person(sample_pos_tag) == 'third person'):
                        verb = token[0]
                        verb = WordNetLemmatizer().lemmatize(verb,'v')
                        
                        length = len(verb)
                        if (verb == 'have'):
                            verb = 'has'
                        elif (verb == 'go'):
                            verb = 'goes'
                        elif (verb == 'do'):
                            verb = 'does'
                        elif (verb[length-2:] == 'ch'):
                            verb = verb + 'es'
                        elif (verb[length-1:] == 's'):
                            verb = verb + 'es'
                        elif (verb[length-2:] == 'sh'):
                            verb = verb + 'es'
                        elif (verb[length-1:] == 'x'):
                            verb = verb + 'es'
                        elif (verb[length-1:] == 'z'):
                            verb = verb + 'es'
                        elif (verb[length-1:] == 'y'):
                            verb = verb[:length-1] + 'ies'
                        else:
                            verb = verb + 's'
                        
                        Perturbed_sample += verb + ' '
                        is_sample_perturbed = True
                        
                    elif (is_third_person(sample_pos_tag) == 'not third person'):
                        verb = token[0]
                        verb = WordNetLemmatizer().lemmatize(verb,'v')
                        
                        Perturbed_sample += verb + ' '
                        is_sample_perturbed = True
                
                else:
                    Perturbed_sample += token[0] + ' '
                    
            elif (remove_negation == True):
                if (token[0] in ('not', "n't")): #----- removing not after do or does
                    Perturbed_sample += ""
                    remove_negation = False
                    
            elif (can_change_basic_form == False):
                if (token[1] == 'VB'): #----- do not change basic form
                    verb = token[0]
                    Perturbed_sample += verb + ' '
                    can_change_basic_form = True
                    
        
        
        print('Perturbed sample:', Perturbed_sample)
        
        if (is_sample_perturbed == True):
            return Perturbed_sample
        else:
            return sample_text





