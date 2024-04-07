import argparse
import os
from ruamel.yaml import *
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import math
from models.model_vqa import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from ExperienceReplay import ExperienceReplayMemory

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer

from vqaTools.vqaEval import VQAEval
from vqaTools.vqa import VQA

from ewc import EWC

import wandb

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def avg_distance(coords):
    total_dist = []
    for i in range(coords.shape[0]):
        individual_dist = []
        for j in range(coords.shape[0]):
            dist = 0
            for k in range(coords.shape[1]):
                dist += (coords[i][k] - coords[j][k]) ** 2
            dist = math.sqrt(dist)
            individual_dist.append(dist)
        total_dist.append(sum(individual_dist)/len(individual_dist))

    return sum(total_dist)/len(total_dist)

words = ['is', 'what', 'have', 'why', 'where', 'how', 'are', 'can']

def train_prompt(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, task, args, arg_opt, ewc):
    model.eval()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    answer_list = data_loader.dataset.answer_list

    total_text_diff = 0
    total_img_diff = 0
    total_image = None
    total_question = None
    print(" in train prompt")
    for i,(image, question, answer, weights, n) in enumerate(data_loader):
       
        if i == 0:
            total_image = image
            total_question = question
        else:
            total_image = torch.cat([total_image,image],dim=0)
            total_question.extend(question)
    
    image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)   
        
    question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 

    
    for i in range(10):
        text_diff, img_diff = model.update_cluster(image,question_input,task)

        total_text_diff = text_diff
        total_img_diff = img_diff

        print("image prompt different is " + str(total_img_diff) + " and text different is " + str(total_text_diff))


        

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, task, args, arg_opt, ewc, replay_memory):
    # train
    model.train()  
    model.to("cuda")
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    answer_list = data_loader.dataset.answer_list

    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)   
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        idxs = None
        if args.cascade_prompt:
            idxs = []
            for q in question:
                find = False
                for w in words:
                    if w in q:
                        idxs.append(words.index(w))    
                        find = True
                        break
                if not find:
                    idxs.append(8)

        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
        loss, updated = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights, task=task, answer_gt=answer, answer_list=answer_list, word_idx = idxs)        


        if args.ER and task != 'a':
            if (i + 1) % 100 == 0:
                sampled_replay_task = replay_memory.sample_replay_task()
                replay_loss = replay_memory.run_replay_step(task_key=sampled_replay_task, model=model, tokenizer=tokenizer, answer_list=answer_list, optimizer=optimizer)
                loss += replay_loss
        if ewc is not None:
            ewc_loss = ewc.compute_ewc_loss(model)[1]
            loss += ewc_loss * 0.1
        #update optimizer if new prompt added
        
        #optimizer = create_optimizer(arg_opt, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        #if updated:
        #    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)

        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size) 
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, task, args) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    accuracy = []
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        
    for n, (image, question, question_id, answers) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)         
        # get caption
        '''samples = {'image':image, 'text_input':question}
        samples = model.module.caption_model.forward_itm(samples=samples)
        samples = model.module.caption_model.forward_cap(samples=samples, num_captions=1, num_patches=20)
        for i in range(len(question)):
            question[i] = question[i] + 'Caption:' + samples['captions'][i][0]'''

        idxs = None
        if args.cascade_prompt:
            idxs = []
            for q in question:
                find = False
                for w in words:
                    if w in q:
                        idxs.append(words.index(w))    
                        find = True
                        break
                if not find:
                    idxs.append(8)

        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)  

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'], task=task, word_idx = idxs)      
        answer = []
        answer_gt = answers[0]
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = ques_id     
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]}) 
            answer.append(data_loader.dataset.answer_list[topk_id[pred]]) 
        
        for a, a_gt in zip(answer, answer_gt):
            #print(str(a) + ' ' + str(a_gt))
            if a == a_gt:
                accuracy.append(1)
            else:
                accuracy.append(0)

    print("the accuracy of dataset " + str(task) + " is " + str(sum(accuracy)/len(accuracy)))

    return result, accuracy


def eval_accuracy(anno_file, ques_file, res_file):
    vqa = VQA(anno_file, ques_file)
    vqaRes = vqa.loadRes(res_file, ques_file)
    vqaEval = VQAEval(vqa,vqaRes, n=2)


def main(args, config):
    #if args.distributed:
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    ewc = None
    replay_memory = None
    if args.ewc:
        ewc = EWC()
    if args.ER:
        replay_memory = ExperienceReplayMemory()
    print("cascade prompt is " + str(args.cascade_prompt))
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")

    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer, l2p=args.l2p, EG_prompt=args.EG_prompt, cascade_prompt=args.cascade_prompt, DualKeyPrompt=args.dual_key_prompt, Dual_Cluster=args.dual_cluster_prompt, ewc=ewc, Dual_Prompt=args.DualPrompt)
    model = model.to(device)   


    total_params = sum(p.numel() for p in model.text_decoder.cls.predictions.parameters())

    print(" the total parameter for cls is " + str(total_params))
    if args.EG_prompt or args.dual_key_prompt or args.l2p or args.dual_cluster_prompt or args.DualPrompt:
        print("freeze model")
        for name, param in model.named_parameters():
            if not 'text_decoder.cls.predictions' in name and not 'caption_model' in name:
                param.requires_grad = False  
    else:
         for name, param in model.named_parameters():
            if 'text_encoder' in name or 'visual_encoder' in name:
                param.requires_grad = False  

  
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cuda') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped   
        
        if False:# not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
                
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.','')         
                    state_dict[encoder_key] = state_dict[key] 
                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)    
                if 'text_encoder' in key:                
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num<6:
                            del state_dict[key]  
                            continue
                        else:
                            decoder_layer_num = (layer_num-6)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)     
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder','text_decoder')  
                    state_dict[decoder_key] = state_dict[key]     

                    del state_dict[key]                
                
        msg = model.load_state_dict(state_dict,strict=False)  
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  

    
    # add prompt functions
    model.add_function()

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)     
    model_without_ddp = model


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module    
    
    
    print("Start training")
    start_time = time.time()

    print("Creating vqa datasets")
    datasets = create_dataset('clove_scene', config)   
    
    
    subtasks=['a','b','c','d','e','f']
    test_loaders = []
    for t in range(len(subtasks)):
       
        sub_datasets = (datasets[0][t],datasets[1][t])
        
        eWc = None
        if args.ewc and t > 0:
          eWc = model.ewc
        else:
            eWc = None

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler(sub_datasets, [True, False], num_tasks, global_rank)         
        else:
            samplers = [None, None]
        
        train_loader, test_loader = create_loader(sub_datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 
       
        test_loaders.append(test_loader)
       
        if args.dual_cluster_prompt:
            train_prompt(model, train_loader, optimizer, tokenizer, None, warmup_steps, device, lr_scheduler, config, subtasks[t], args, arg_opt, eWc)
        
        
        for epoch in range(start_epoch, max_epoch):
            if epoch>0:
                lr_scheduler.step(epoch+warmup_steps)  
            
            if not args.evaluate:
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)

                train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, subtasks[t], args, arg_opt, eWc, replay_memory) 
               

                _,_ = evaluation(model, test_loader, tokenizer, device, config, subtasks[t],args)

            if args.evaluate:
                break
            
            if args.distributed:
                if utils.is_main_process():               
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'epoch': epoch,
                                }                
                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")                        
                                
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

            if args.distributed:
                dist.barrier()   
        # evaluate test based on all previous val dataset

        if args.ewc and t < len(subtasks)-1:

            eWc=model.ewc
            eWc.save_task_parameters(task_key=subtasks[t],
                                        model=model,
                                        optimizer=optimizer,
                                        dataloader=train_loader,
                                        tokenizer=tokenizer
            )

        if args.ER:
            replay_memory.add_task_memory_buffer(args=args,
                                                task_key=subtasks[t],
                                                train_dataset=train_loader.dataset,
                                                memory_percentage=0.01,
                                                sampling_strategy='random')


        if args.distributed:
            model.module.EG_classifiers[subtasks[t]] = copy.deepcopy(model.module.text_decoder.cls)
        else:
            model.EG_classifiers[subtasks[t]] = copy.deepcopy(model.text_decoder.cls)


        if args.EG_prompt:
           
            for _, param in model.module.Prompt_E[subtasks[t]].named_parameters():
                param.require_grad = False
            for _, param in model.module.EG_classifiers[subtasks[t]].named_parameters():
                param.require_grad = False

        if args.distributed:
            model.module.EG_classifiers[subtasks[t]] = copy.deepcopy(model.module.text_decoder.cls)
        else:
            model.EG_classifiers[subtasks[t]] = copy.deepcopy(model.text_decoder.cls)

        #model.module.text_decoder.cls.load_state_dict()

        if t <= 5:
            total_accuracy = []
            i = 0
            for task in range(t+1):
                sub_task = (datasets[0][task],datasets[1][task])
                if args.distributed:
                    num_tasks = utils.get_world_size()
                    global_rank = utils.get_rank()            
                    samplers = create_sampler(sub_task, [True, False], num_tasks, global_rank)         
                else:
                    samplers = [None, None]

                train_loader, test_loader = create_loader(sub_task,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 

                vqa_result, acc = evaluation(model, test_loader, tokenizer, device, config, subtasks[task],args)   
                i += 1
                total_accuracy.extend(acc)  
            if t != 0:
                print("the total avg accuracy is " + str(sum(total_accuracy)/len(total_accuracy)))   

    result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)


    #anno_file = '/project/rostamim_919/caiyulia/data/CLOVE/json/scene/scene_a#ShopAndDining_val.json'
    #eval_accuracy(anno_file,anno_file,result_file)
                     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml') 
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--l2p', default=False, type=bool) #l2p type of prmompt
    parser.add_argument('--EG_prompt', default=False, type=bool) #genral prompt + specific prmopt
    parser.add_argument('--cascade_prompt', default=False, type=bool) # prompt of word followed by img 
    parser.add_argument('--dual_key_prompt', default=False, type=bool) # prompt for dual key prompt
    parser.add_argument('--dual_cluster_prompt', default=False, type=bool) # propmt for dual key with clustering
    parser.add_argument('--ewc', default=False, type=bool) # use ewc or not
    parser.add_argument('--ER', default=False, type=bool)
    parser.add_argument('--DualPrompt', default=False, type=bool)
    args = parser.parse_args()

    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)