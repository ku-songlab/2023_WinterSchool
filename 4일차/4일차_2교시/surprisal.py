#!/usr/bin/env python

"""surprisal.py
Calculate the surprisal.
"""

import torch
import re
import sys
import os
import statistics

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch import device
from transformers import AdamW, BertConfig, BertTokenizer, BertForMaskedLM

from sys import path
path.append(os.path.abspath(os.getcwd())+"/bertviz_lin")
#from bertviz_lin import attention, visualization 
#from bertviz_lin.pytorch_pretrained_bert import BertModel, BertForTokenClassification
#for confusion score estimation

class AttentionGenerator:
# Code from Lin, Tan, & Frank (2019)
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def get_data(self, sentence_a):
        tokens_tensor, token_type_tensor, tokens_a = self._get_inputs(sentence_a)
        attn = self._get_attention(tokens_tensor, token_type_tensor)
        return tokens_a, attn

    def _get_inputs(self, sentence_a):
        tokens_a = self.tokenizer.tokenize(sentence_a)
        tokens_a_delim = ['[CLS]'] + tokens_a + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens_a_delim)
        tokens_tensor = torch.tensor([token_ids])
        token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim)])
        return tokens_tensor, token_type_tensor, tokens_a_delim

    def _get_attention(self, tokens_tensor, token_type_tensor):
        _, _, attn_data_list = self.model(tokens_tensor, token_type_ids=token_type_tensor)
        attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
        return attn_tensor.data.numpy()

def find_indices(pattern, string, ignore=False): 
    """Find the indices
    Args:
        pattern (str): a pattern to find
        string (str): a string to search
        ignore (bool): case matching

    Returns:
        [(int start, int end)]: list of the instances. The instance starts at start and end before end.
    """
    
    if not ignore:
        r = []
        for i in re.finditer(pattern, string):
            r.append((i.start(), i.end()))
        return r
    else:
        r = []
        for i in re.finditer(pattern, string, flags=re.IGNORECASE):
            r.append((i.start(), i.end()))
        return r

class ETRI_word:
    """A word as represented by ETRI

    Attributes:
        lemma: the lemma of the word
        tag: the tag assigned to the lemma
        concat: concatnated form of the lemma and the tag
    """

    lemma=""
    tag=""
    concat=""
    
    def __init__(self, lemma, tag):
        self.lemma = lemma
        self.tag = tag
        self.concat = lemma + "/" + tag + "_"

    def print_content(self):
        """Print out the instance.
        """
        print("lemma: {} | tag: {} | concat: {}".format(self.lemma, self.tag, self.concat))
        

def analyze_etri_morph(text, key):
    """Perform a morphological analysis with ETRI

    Args:
        text (str): the text to analyze
        key (str): the API key for ETRI
    
    Returns:
        [ETRI_word]: list of morphonologically analyzed words

    Raises:
        RuntimeError: abnormal behavior from ETRI
    """
    import json, subprocess, urllib3

    URL = "https://aiopen.etri.re.kr:8443/WiseNLU"

    request_JSON = {
        "access_key": key,
        "argument": {
            "text": text,
            "analysis_code": "morp"
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        URL,
        headers = {"Content-Type": "application/json; cahrset=UTF-8"},
        body=json.dumps(request_JSON)
    )

    if response.status != 200:
        raise RuntimeError("ETRI did not respond with 200 (success)")

    response = json.loads(response.data)

    if response['result'] != 0:
        raise RuntimeError("ETRI could not morphologically analyze the phrase.")

    response = response["return_object"]["sentence"][0]["morp"]

    resulting_words = []

    for r in response:
        new_word = ETRI_word(r["lemma"], r["type"])
        resulting_words.append(new_word)
    
    return resulting_words

def bert_token_surprisal(text, keywords, mask_model, tokenizer, device, printing = True, add_period = True, is_etri = False, key = ""):
    """Show the surprisal on the token

    Args:
        text (str): experimental texts
        keywords ([str]): keywords to input in the [MASK]
        maks_model (BERTForMaskedLM): the BERTMaskedLM
        tokenizer (BERTTokenizer): the tokenizer to be used
        device (Device): GPU(CUDA) or CPU
        printing (Bool): Whether to print out the result or not
        add_period (Bool): Whether to automatically add a period or not at the end of a sentence
        is_etri (Bool): True if the BERT is ETRI's KorBERT
        key (str): API key for ETRI
    Returns:
        [(string, float)]: list of sentences completed with keyword and its surprirsal
    Raises:
        RuntimeError: When API key is not available despite is_etri
        RuntimeError: When recognized words does not match with the surprisal
    """
    # Check the key

    if is_etri and key == "":
        raise RuntimeError("An API Key is needed for ETRI KorBERT.")

    # Tokenize
    
    tokenized_lines=[]
    lower = lambda x: " ".join(a if re.match(r".*\[MASK\].*", a) else a.lower() for a in x.split())
    if is_etri:
        for line in text.split("\n"):
            if line != "":
                line = lower(line)
                line = line.replace("[MASK]", " MASK ") # Make the [MASK] compatible with ETRI KorBERT
                line = analyze_etri_morph(line, key)
                resulting_words = [word.concat for word in line]
                # Add special tokens
                resulting_words.insert(0, "[CLS]")
                resulting_words.insert(len(resulting_words), "[SEP]")
                tokenized_lines.append(resulting_words)

    else:
        for line in text.split("\n"):
            if line != "":
                line = lower(line)
                line = "[CLS]" + line + "[SEP]" # Add special tokens
                tokenized_line = tokenizer.tokenize(line)
                if add_period and tokenized_line[-2] not in [".", "?", "!"]: # If no punctuation
                    # Force add a period at the end of the sentence
                    tokenized_line.insert(-1, ".")
                tokenized_lines.append(tokenized_line)

    result="Experimenting sentences:\n"
    result= result + text+ "\n"
    result=result+"Experimenting words:"+" "
    for keyword in keywords:
        keyword = keyword.lower()
        result=result+keyword+" "
    result=result+"\n\n"

    pairs = []
    
    # Compute Surprisal
    for line in tokenized_lines:
        result=result+"Tokenization result: \n"
        result=result+str(line)+"\n"
        if is_etri:
            masked_index = line.index('MASK/SL_')
            indexed_tokens = []
            for w in line:
                try: 
                    indexed_tokens.append(tokenizer.vocab[w])
                except KeyError:
                    indexed_tokens.append(1) # [UNK] = 1
        else:
            masked_index = line.index('[MASK]')
            indexed_tokens = tokenizer.convert_tokens_to_ids(line)
        tens = torch.LongTensor(indexed_tokens).unsqueeze(0)
        if device == torch.device("cuda"):
            tens = tens.to('cuda')
            mask_model.to('cuda')
        else:
            tens = tens.to('cpu') 
            mask_model.to('cpu')
        output = mask_model(tens)
        res = output[0]
        res = res[0,masked_index]
        res = F.softmax(res, -1)
        word_ids = tokenizer.convert_tokens_to_ids(keywords)
        recognized_words = tokenizer.convert_ids_to_tokens(word_ids)
        scores = res[word_ids]
        surprisals =  -1 * torch.log2( scores )
        surprisals = surprisals.cpu()
        surprisals = surprisals.detach().numpy()
        
    # Print the result
        result=result+"\nRecognized Words:"+" "
        resulting_sentences = []
        for word in recognized_words:
            result=result+word+" "
            resulting_sentence = line
            resulting_sentence = ' '.join(resulting_sentence)
            resulting_sentence = resulting_sentence.replace(' ##', '')
            resulting_sentence = resulting_sentence.replace("[CLS]", "")
            resulting_sentence = resulting_sentence.replace("[SEP]", "")
            resulting_sentence = resulting_sentence.replace(" .", ".") 
            resulting_sentences.append(resulting_sentence)

        result=result+"\n"
        if len(recognized_words) == len(surprisals):
            for i in range(len(surprisals)):
                result=result+ str(surprisals[i]) +"  "+ recognized_words[i]+"\n"
                pairs.append((resulting_sentences[i], recognized_words[int(i%len(recognized_words))], surprisals[i]))
            result=result+"\n\n"
        else:
            raise RuntimeError("Recocgnized words do not match up with the surprisals.")
    
    if printing: 
        print(result)

    return pairs

def bert_sentence_surprisal(sentence, mask_model, tokenizer, device, printing = True, add_period = True, is_etri = False, key = ""):
    """Show the surprisal on the sentence

    Args:
        sentence (str): experimental sentence
        maks_model (BERTForMaskedLM): the BERTMaskedLM
        tokenizer (BERTTokenizer): the tokenizer to be used
        device (Device): GPU(CUDA) or CPU
        printing (Bool): Whether to print out the resut or not
        add_period (Bool): Whether to automatically add a period or not at the end of a sentence
        is_etri (Bool): True if the BERT is ETRI's KorBERT
        key (str): API key for ETRI
    Returns:
        [(string, float)]: list of sentences completed with keyword and its surprirsal
    Raises:
        RuntimeError: When API key is not available despite is_etri
        RuntimeError: When recognized words does not match with the surprisal
    """ 
    sentence = sentence.lower()

    if is_etri:
        line = analyze_etri_morph(sentence, key)
        resulting_words = [word.concat for word in line]
        tokenized_sentence = resulting_words
    else:
        tokenized_line = tokenizer.tokenize(sentence)
        tokenized_sentence = tokenized_line
    
    result="Experimenting sentence:"
    result = result + sentence + "\n"
    result = result+"\n\n"

    surprisals = []

    for i in range(len(tokenized_sentence)):
        temp = tokenized_sentence[i]
        if temp not in ['[CLS]', '[SEP]', '.', '!', '?']:
            tokenized_sentence[i] = '[MASK]'
            text = " ".join(tokenized_sentence)
            token_surprisal = bert_token_surprisal(text, [temp], mask_model, tokenizer, device, False, add_period, is_etri, key)
            surprisals.append(token_surprisal[0][2])
            tokenized_sentence[i] = temp

    surprisal_avg = np.mean(surprisals)

    result = result + "Average suprisal for the sentence \t"+str(surprisal_avg)+"\n\n"
    if printing:
        print(result)
    
    return [(sentence, surprisal_avg)]
        


def confusion_score(sentence, model, tokenizer):
    # Code from Lin, et al. (2019)
    """Calculate the confusion score from the file.
    Args:
        sentence(str): sentence to calculate the confusion score
        model (BertForTokenClassification): a BERT model
        tokenizer (BertTokenizer): a tokenizer to be used

    Returns:
        [(string, float)]: sentence and corresponding confusion score

    """
    # Code inspired by Lin, Tan, & Frank (2019)
    n_layers = model.config.num_hidden_layers
    attention_generator = AttentionGenerator(model, tokenizer)

    bces = np.empty([1, n_layers])
    sentence = sentence.lower()
    line = sentence.strip().split('\t')
    sentence, source, *target_groups = line
    source_idx = int(source) + 1
    target_groups_idx = []
    for group in target_groups:
        str_idxes = group.strip().split()
        target_groups_idx.append(list(map(lambda s: int(s) + 1, str_idxes))) # offset for [CLS]
    tokens, attn = attention_generator.get_data(sentence)
    source_attn = np.empty([n_layers])
    for layer in range(n_layers):
        layer_attn = attn[layer][0] # Get layer attention (assume batch size = 1), shape = [num_heads, seq_len, seq_len]
        head_avg = np.mean(layer_attn, axis=0) # shape = [seq_len, seq_len]
        grouped_attn = [head_avg[source_idx, group].sum() for group in target_groups_idx]
        grouped_attn /= sum(grouped_attn)
        source_attn[layer] = grouped_attn[0] 
    
    bce = -np.log2(source_attn)
    bces[0] = bce

    avg_bce = np.mean(bces, axis=0)
    avg_summed_bce = np.sum(avg_bce)

    confusion_score = avg_summed_bce/len(avg_bce)

    return [(sentence, confusion_score)]

def confusion_score_batch(file_path, model, tokenizer):
    # Code from Lin, et al. (2019)
    """Calculate the confusion score from the file.
    Args:
        file_path(str): path to the file
        model (BertForTokenClassification): a BERT model
        tokenizer (BertTokenizer): a tokenizer to be used

    Returns:
        float: the calcuated mean of confusion score

    """
    # Code from Lin, Tan, & Frank (2019)
    n_layers = model.config.num_hidden_layers
    attention_generator = AttentionGenerator(model, tokenizer)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    bces = np.empty([len(lines), n_layers])
    for idx, line in tqdm(enumerate(lines)):

        # preprocess lines
        line = line.lower()
        line = line.strip().split('\t')
        sentence, source, *target_groups = line
        source_idx = int(source) + 1 # offset for [CLS]
        target_groups_idx = []
        for group in target_groups:
            str_idxes = group.strip().split()
            target_groups_idx.append(list(map(lambda s: int(s) + 1, str_idxes))) # offset for [CLS]

        tokens, attn = attention_generator.get_data(sentence)
        source_attn = np.empty([n_layers])
        for layer in range(n_layers):
            layer_attn = attn[layer][0] # Get layer attention (assume batch size = 1), shape = [num_heads, seq_len, seq_len]
            head_avg = np.mean(layer_attn, axis=0) # shape = [seq_len, seq_len]
            grouped_attn = [head_avg[source_idx, group].sum() for group in target_groups_idx]
            grouped_attn /= sum(grouped_attn)
            source_attn[layer] = grouped_attn[0]

        bce = -np.log2(source_attn)
        bces[idx] = bce

    corpus_avg_bce = np.mean(bces, axis=0)
    corpus_avg_summed_bce = np.sum(corpus_avg_bce)

    confusion_score = corpus_avg_summed_bce/len(corpus_avg_bce)

    return confusion_score