import argparse
import random
import logging
from typing import List, Dict
import torch
from torch import nn
from torch.optim import AdamW
from itertools import chain

logger = logging.getLogger(__name__)


class ExperienceReplayMemory:

    def __init__(self):
        '''
        Initializes ER memory with empty memory buffer dict
        '''
        self.memory_buffers = {}
        self.pt_pos_emb = None

    def add_task_memory_buffer(self, 
                               args=None, 
                               task_key=None,
                               train_dataset=None,
                               memory_percentage=None, 
                               sampling_strategy=None):
        '''
        Creates a memory buffer for new task
        '''

        task_buffer = TaskMemoryBuffer(args, task_key, train_dataset, memory_percentage, sampling_strategy)
        self.memory_buffers[task_key] = task_buffer

    def do_replay(self) -> bool:
        '''
        Return true if there are any tasks in the memory to do replay on, else False
        '''
        return True if len(self.memory_buffers) > 0 else False

    def sample_replay_task(self) -> str:
        '''
        Samples a previous task at random
        '''
        previous_tasks = list(self.memory_buffers.keys())
        #ran_int = random.randint(0,1)
        #if ran_int < 0.3 and 'cocoqa' in previous_tasks:
        #    sampled_previous_task = 'cocoqa'
        #elif ran_int < 0.7:
        #    sampled_previous_task = 'nlvr2'
        #else:
        sampled_previous_task = random.choice(previous_tasks)
        return sampled_previous_task

    def create_optimizer(self, model, task_key):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        lr = 3e-4
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
        return optimizer

    def run_replay_step(self, task_key: str, model: nn.Module, tokenizer: nn.Module, answer_list: list, optimizer: nn.Module) -> torch.Tensor:
        '''
        Performs a single training step on previous task, by sampling a batch from task bugger
        '''
        task_buffer = self.memory_buffers[task_key]
        #optimizer = self.create_optimizer(model,task_key)
        image, question, answer, weights = task_buffer.sample_replay_batch()
       
        image, weights = image.to('cuda',non_blocking=True), weights.to('cuda',non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to('cuda') 

        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to('cuda') 
       
        replay_loss, _= model(image, question_input, answer_input, train=True, alpha=0.4, k=[10 for i in range(len(question))], weights=weights, task=None, answer_gt=answer, answer_list=answer_list, word_idx = None)        


        logger.info("{} replay step: loss = {:.5f}".format(task_key, replay_loss.item()))
        return replay_loss

class TaskMemoryBuffer:

    '''
    Buffer of training examples that can be used for replay steps
    '''
    def __init__(self, 
                 args=None, 
                 task_key=None, 
                 train_dataset=None,
                 memory_percentage=None, 
                 sampling_strategy=None):

        '''
        Creates a memory buffer for new task, which samples a small percentage of training data for experience replay
        '''

        self.task_key = task_key

        self.dataset = train_dataset
        self.batch_size = 16

        self.memory_percentage = 0.01                # Percent of training samples to store in memory
        assert self.memory_percentage < 1.0
        self.memory_size = int(memory_percentage*len(self.dataset))     # Number of training samples that are stored in memory
        self.sampling_strategy = sampling_strategy
        assert sampling_strategy in ['random']                      # Only random sampling for memory buffer implemented so far

        if self.sampling_strategy == 'random':
            train_idxs = list(range(len(self.dataset)))
            self.memory_idxs = random.sample(train_idxs, self.memory_size)

        elif self.sampling_strategy == 'random-balanced':
            raise NotImplementedError("Label-balanced sampling of replay memory is not yet implemented!")

        #logger.info("Created {} replay memory buffer, with {} samples in the memory".format(self.task_name, len(self.memory_idxs)))

    def __len__(self):
        return len(self.memory_idxs)

    def sample_replay_batch(self) -> Dict:

        sampled_instances = random.sample(self.memory_idxs, self.batch_size)
     
        image = torch.stack([self.dataset[i][0] for i in sampled_instances],dim=0)
        question = [self.dataset[i][1] for i in sampled_instances]
        answer = [self.dataset[i][2] for i in sampled_instances]
        answer = list(chain.from_iterable(answer))
        weight = torch.cat([torch.Tensor(self.dataset[i][3]) for i in sampled_instances],dim=0)
        return image, question, answer, weight
