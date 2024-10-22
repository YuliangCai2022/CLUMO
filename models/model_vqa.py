from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel

import torch
from torch import nn
import torch.nn.functional as F
from prompt import  DualKeyPrompt_cluster
import numpy as np
import copy
from copy import deepcopy

CLOVE_tasks = ['a','b','c','d','e','f']

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,    
                 l2p = False, 
                 EG_prompt = False,
                 cascade_prompt = False,
                 DualKeyPrompt = False,
                 Dual_Cluster = False,
                 Dual_Prompt = False,
                 ewc = None,
                 SPrompt = False,
                 ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer 
        self.distill = config['distill']
        self.l2p = l2p
        self.EG_prompt = EG_prompt
        self.Dual_Cluster = Dual_Cluster
        self.DualKeyPrompt = DualKeyPrompt
        self.cascade_prompt = cascade_prompt
        self.SPrompt = SPrompt
        self.Dual_Prompt = Dual_Prompt
        self.prompt_classifier = None
        self.prev_cls = None
        self.prompt_classifier_dict = {}
        self.prompt_classifiers = nn.ModuleDict(self.prompt_classifier_dict)
        

        self.caption_model = None
        self.EG_classifiers = dict()
        self.loss = nn.CrossEntropyLoss()
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False) 
            
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)    
        # text decoder output 768 x 30524


        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))             
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)   
            self.text_decoder_m = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)   
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_decoder,self.text_decoder_m],
                               ]
            self.copy_params() 
            self.momentum = 0.995

        self.ewc = ewc
        print("ewc is none? " + str(self.ewc == None))

        
    def update_prev_cls(self):
        self.prev_cls = deepcopy(self.text_decoder.cls.predictions)
    def add_function(self):


        if self.Dual_Cluster:
            print(" in dual key prompt model")
            self.prompt_length = 10 #
            embed_dim = 768
            embedding_key = 'cls'
            prompt_init = 'uniform'
            prompt_pool = True
            prompt_key = True
            pool_size = 30 # 5 for initialization
            top_k = 1 #1
            self.top_k = top_k
            batchwise_prompt = True
            prompt_key_init = 'uniform'
            self.Prompt_pools_dict = {}
            self.Prompt_pools = nn.ModuleDict(self.Prompt_pools_dict)
            self.general_prompt_length = 10
            for i in range(len(CLOVE_tasks)):
                self.Prompt_pools[CLOVE_tasks[i]]=DualKeyPrompt_cluster(length=self.prompt_length, embed_dim=embed_dim, embedding_key=embedding_key, prompt_init=prompt_init,
                        prompt_pool=prompt_pool, prompt_key=prompt_key, image_key_size=3, text_key_size=3, top_k=top_k, batchwise_prompt=batchwise_prompt,
                        prompt_key_init=prompt_key_init,general_prompt_length=self.general_prompt_length).to("cuda")
                classifier = copy.deepcopy(self.text_decoder.cls)
                self.prompt_classifiers[CLOVE_tasks[i]] = classifier

    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def update_cluster(self, image, quesiton, task, both_mod):
        image_embeds = torch.mean(self.visual_encoder(image),dim=1)
       
        #image_embeds = self.visual_encoder(image)[:,0,:]
        if both_mod:
            text_embeds = torch.mean(self.text_encoder.embeddings(input_ids=quesiton.input_ids,
                        position_ids=None,
                        token_type_ids=torch.zeros(quesiton.input_ids.size(), dtype=torch.long, device='cuda'),
                        inputs_embeds=None,
                        past_key_values_length=0,),dim=1)
           

            
        image_cat = [[] for _ in range(self.Prompt_pools[task].image_key_size)]
        if both_mod:
            text_cat = [[] for _ in range(self.Prompt_pools[task].text_key_size)]
        image_different = 0
        text_different = 0
        for i in range(image_embeds.shape[0]):
            max_dist = -1
            max_idx = 0
            for j in range(self.Prompt_pools[task].image_key_size):
                #dist = torch.matmul(image_embeds[i].to('cuda'), self.Prompt_pools[task].img_prompt_key[j].T.to("cuda"))
                dist = (image_embeds[i].to('cuda')- self.Prompt_pools[task].img_prompt_key[j].to("cuda")).pow(2).sum(0).sqrt()
                if dist > max_dist:
                    max_dist = dist
                    max_idx = j
            image_cat[max_idx].append(image_embeds[i].reshape(1,-1))
            #image_different += min_dist
        #image_different /= image_embeds.shape[0]
        if both_mod:
            for i in range(text_embeds.shape[0]):
                max_dist = -1
                max_idx = 0
                for j in range(self.Prompt_pools[task].text_key_size):
                    #dist = torch.matmul(text_embeds[i].to('cuda'), self.Prompt_pools[task].text_prompt_key[j].T.to('cuda'))
                    dist = (text_embeds[i].to('cuda')-self.Prompt_pools[task].text_prompt_key[j].to('cuda')).pow(2).sum(0).sqrt()
                    if dist > max_dist:
                        max_dist = dist
                        max_idx = j
                text_cat[max_idx].append(text_embeds[i].reshape(1,-1))
            #text_different += min_dist
        #text_different /= text_embeds.shape[0]
        image_different = 0
        text_different = 0
        for i in range(len(image_cat)):
            if len(image_cat[i]) > 0:
                image_cat[i] = torch.mean(torch.cat(image_cat[i],dim=0),dim=0)
                #image_different += torch.matmul(image_cat[i].to('cuda'), self.Prompt_pools[task].img_prompt_key[i].T.to('cuda'))
                image_different += (image_cat[i].to('cuda')-self.Prompt_pools[task].img_prompt_key[i].to("cuda")).pow(2).sum(0).sqrt()
                self.Prompt_pools[task].img_prompt_key[i] = image_cat[i]
        if both_mod:
            for i in range(len(text_cat)):
                if len(text_cat[i]) > 0:
                    text_cat[i] = torch.mean(torch.cat(text_cat[i],dim=0),dim=0)
                    #text_different +=  torch.matmul(text_cat[i].to('cuda'), self.Prompt_pools[task].text_prompt_key[i].T.to('cuda'))
                    text_different += (text_cat[i].to('cuda')- self.Prompt_pools[task].text_prompt_key[i].to('cuda')).pow(2).sum(0).sqrt()
                    self.Prompt_pools[task].text_prompt_key[i] = text_cat[i]

        return text_different, image_different



    def forward(self, image, quesiton, answer=None, alpha=0, k=None, weights=None, train=True, task=None, answer_gt=None, answer_list=None, word_idx = None, reduce_pool=None):
        reduce_sim = 0
        reduce_diff = 0
        prompt_updated = False
        image_embeds = None
        
        image_embeds = self.visual_encoder(image).to('cuda') 
        if self.Dual_Cluster:
            
            text_embeds = self.text_encoder.embeddings(input_ids=quesiton.input_ids,
                    position_ids=None,
                    token_type_ids=torch.zeros(quesiton.input_ids.size(), dtype=torch.long, device='cuda'),
                    inputs_embeds=None,
                    past_key_values_length=0,).to('cuda')
        idx = None

       
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

    
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''          
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

                
            question_output = self.text_encoder(quesiton.input_ids, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    

            if self.Dual_Cluster:
                res = self.Prompt_pools[task](text_embed=text_embeds, img_embed=image_embeds, prompt_mask=None, cls_features=None, add_prompt=True)
                prompt = res['batched_prompt']
                reduce_sim = res['reduce_sim']
                reduce_diff = res['reduce_diff']
                #self.prompt_classifier = self.prompt_classifiers[str(res['prompt_idx'][0][0].item())]
                idx = res['prompt_idx']
                question_output.last_hidden_state=torch.cat([prompt, question_output.last_hidden_state], dim=1)
                prompt_updated = res['modified']
                for i in range(image_embeds.shape[0]):
                    reduce_pool[int(res['text_idx'][i][0].to("cpu"))].append(torch.mean(text_embeds[i],dim=0).reshape(1,-1))


            question_states = []                
            question_atts = []  
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]]*n
                if self.Dual_Cluster:
                    size = 1
                    if self.top_k > 1:
                        size = 3
                    question_atts += [torch.cat([torch.ones(self.prompt_length*size+self.general_prompt_length).to('cuda'), quesiton.attention_mask[b]])]*n 
                else:
                    question_atts += [quesiton.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     
            distill_loss = 0
            if self.distill:                    
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    question_output_m = self.text_encoder_m(quesiton.input_ids, 
                                                            attention_mask = quesiton.attention_mask, 
                                                            encoder_hidden_states = image_embeds_m,
                                                            encoder_attention_mask = image_atts,                             
                                                            return_dict = True)    

                    question_states_m = []                
                    for b, n in enumerate(k):
                        question_states_m += [question_output_m.last_hidden_state[b]]*n
                    question_states_m = torch.stack(question_states_m,0)    

                    logits_m = self.text_decoder_m(answer.input_ids, 
                                                   attention_mask = answer.attention_mask, 
                                                   encoder_hidden_states = question_states_m,
                                                   encoder_attention_mask = question_atts,                                  
                                                   return_logits = True,
                                                  )                       

                
                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  soft_labels = F.softmax(logits_m,dim=-1),
                                                  alpha = alpha,
                                                  reduction = 'none',
                                                 )   
            else:
                
                #classifier = self.EG_classifiers[task]
                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                  classifiers = self.prompt_classifiers,
                                                  classifier_idx = task,
                                                 ) 
                if self.prev_cls is not None:
                    distill_loss = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                  classifiers = self.prompt_classifiers,
                                                  classifier_idx = task,
                                                  prev_cls = self.prev_cls
                                                 )   
                

            loss = weights * answer_output.loss
            if self.Dual_Cluster:
                loss = loss.sum()/image.size(0)  +  distill_loss
            else:
                loss = loss.sum()/image.size(0)
            return loss, prompt_updated
            

        else: 
            question_output = self.text_encoder(quesiton.input_ids, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True)      

            if self.Dual_Cluster:
                res = self.Prompt_pools[task](text_embed=text_embeds, img_embed=image_embeds, prompt_mask=None, cls_features=None)
                prompt = res['batched_prompt']
                reduce_sim = res['reduce_sim']
                #self.prompt_classifier = self.prompt_classifiers[str(res['prompt_idx'][0][0].item())]
                idx = res['prompt_idx']
                question_output.last_hidden_state=torch.cat([prompt, question_output.last_hidden_state], dim=1)
                size = 1
                if self.top_k > 1:
                    size = 3
                quesiton.attention_mask = torch.cat([torch.ones([quesiton.attention_mask.shape[0],self.prompt_length*size+self.general_prompt_length]).to('cuda'), quesiton.attention_mask],dim=1)
            
         

            CL_method = None
            if self.Dual_Cluster:
                CL_method = 'dual_cluster'

            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, quesiton.attention_mask, 
                                                    answer.input_ids, answer.attention_mask, k, task, CL_method) 
            return topk_ids, topk_probs
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k, task, CL_method):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(input_ids=start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none',
                                         classifiers = self.prompt_classifiers,
                                         classifier_idx = task,
                                         CL_method=CL_method)#EG_classifier)    
        

        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none',
                                   classifiers = self.prompt_classifiers,
                                   classifier_idx = task,
                                   CL_method = CL_method)           

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
