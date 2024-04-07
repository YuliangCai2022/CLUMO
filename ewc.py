import argparse
import random
from collections import defaultdict
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple

import torch


class EWC:

    def __init__(self):
        '''
        Initializes an EWC object with EWC parameters and empty dictionaries that will be used for storing model parameters
        '''
        self.fisher_sample_percentage = 0.1
        self.ewc_loss_weight = 0.1
        self.fisher_dict = {}
        self.param_dict = {}
        self.task_keys = []
        self.device = 'cuda'

    def save_task_parameters(self, 
                             task_key=None, 
                             model=None, 
                             optimizer=None,
                             dataloader=None,
                             tokenizer=None,
                             ):
        '''
        Saves model parameters after training on a task, and computes Fisher information matrix
        '''

        self.fisher_dict[task_key] = defaultdict(float)

        # Save model params
        self.param_dict[task_key] = {}
        for name, param in model.named_parameters():
            self.param_dict[task_key][name] = param.data.cpu().clone()
        assert task_key not in self.task_keys
        self.task_keys.append(task_key)

        fisher_sample_size = int(self.fisher_sample_percentage*len(dataloader.dataset))

        model.eval()
        model.to(self.device)
        optimizer.zero_grad()
        num_samples_completed = 0
        # Create fisher matrix
        answer_list = dataloader.dataset.answer_list
        for i,(image, question, answer, weights, n) in enumerate(tqdm(dataloader,)):
            image, weights = image.to(self.device,non_blocking=True), weights.to(self.device,non_blocking=True)   
        
            question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(self.device) 

            answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(self.device) 
            
            alpha = 0.4*min(1,i/len(dataloader))
                
            loss, _ = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights, task=task_key, answer_gt=answer, answer_list=answer_list, word_idx = None)

            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher_dict[task_key][name] += param.grad.data.pow(2).cpu().clone()

            num_samples_completed += len(question)
            if num_samples_completed >= fisher_sample_size:
                break

        for name in self.fisher_dict[task_key].keys():
            self.fisher_dict[task_key][name] /=  num_samples_completed

    def compute_ewc_loss(self, model=None):
        '''
        Randomly samples previous task, and computes EWC loss by comparing model parameters with previous parameters
        '''

        ewc_task_key = random.choice(self.task_keys)
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher_dict[ewc_task_key].keys():
                ewc_param = self.param_dict[ewc_task_key][name].to(self.device)
                fisher_info = self.fisher_dict[ewc_task_key][name].to(self.device)
                ewc_loss += (fisher_info*((param - ewc_param).pow(2))).sum()
        return ewc_task_key, self.ewc_loss_weight*ewc_loss

  