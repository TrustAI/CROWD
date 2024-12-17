import json
import numpy as np
import torch
import string
import spacy
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import math

class WordSubstitude:
    def __init__(self, table, word_embd):
        self.table = table
        self.word_embd = word_embd
        self.table_key = set(list(table.keys()))
        self.exclude = set(string.punctuation)

    def get_perturbed_batch(self, batch, rep):
        num_text = len(batch)
        out_batch_ran = []
        for k in range(rep):
            for i in range(num_text):
                tem_text = batch[i][0].split(' ')
                cln_text = (' '.join(tem_text))
                if tem_text[0]:
                    for j in range(len(tem_text)):
                        if tem_text[j][-1] in self.exclude:
                            tem_text[j] = self.sample_from_table(tem_text[j][0:-1], self.word_embd) + tem_text[j][-1]
                        else:
                            tem_text[j] = self.sample_from_table(tem_text[j], self.word_embd)
                    out_batch_ran.append([str(' '.join(tem_text))])
                else:
                    out_batch_ran.append([str(batch[i][0])])
        return np.array(out_batch_ran)

    def softmax(self, x, temperature = 0.01):
        print('temperature', temperature)
        e_x = np.exp((x - np.max(x)) / temperature) 
        return e_x / e_x.sum(axis=0)

    def beta_distribution(self, sim_score_ls, tem_words):
        mean = np.mean(sim_score_ls)
        variance = np.var(sim_score_ls)
        alpha = mean * (mean * (1 - mean) / variance - 1)
        beta = (1 - mean) * (mean * (1 - mean) / variance - 1)
        sampled_probabilities = np.random.beta(alpha, beta, len(sim_score_ls)) * sim_score_ls
        sampled_probabilities /= sampled_probabilities.sum()
        selected_index = np.argmax(sampled_probabilities)
        selected_word = tem_words[selected_index]
        return selected_word
            
    # sample from most similar word based on designed distribution
    def sample_from_table(self, word, word_embd):
        if word in self.table_key:
            tem_words = self.table[word]['set']
            num_words = len(tem_words)
            orig_word_vector = word_embd[word]
            similarity_score_ls = [None for _ in range(len(tem_words))]
            for index, tem_word in enumerate(tem_words):
                tem_word_vector = word_embd[tem_word]
                similarity_score = cosine_similarity([orig_word_vector], [tem_word_vector])[0][0]
                similarity_score_ls[index] = similarity_score
            sample = self.beta_distribution(similarity_score_ls, tem_words)
            return sample
        else:
            return word
    


        