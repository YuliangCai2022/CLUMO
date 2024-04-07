import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question


class clove_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30, answer_list='', SorF='s', sub_data='a'):
        sub_task_s = dict()
        sub_task_s['a'] = 'scene/scene_a#ShopAndDining_'
        sub_task_s['b'] = 'scene/scene_b#Workplace_'
        sub_task_s['c'] = 'scene/scene_c#HomeOrHotel_'
        sub_task_s['d'] = 'scene/scene_d#Transportation_'
        sub_task_s['e'] = 'scene/scene_e#SportAndLeisure_'
        sub_task_s['f'] = 'scene/scene_f#Outdoors_'

        sub_task_f = dict()
        sub_task_f['a'] = 'funcion/functional_attribute_'
        sub_task_f['b'] = 'function/functional_knowledge_'
        sub_task_f['c'] = 'function/functional_logical_'
        sub_task_f['d'] = 'function/functional_object_'
        sub_task_f['e'] = 'function/functional_relation_'
        sub_task_f['f'] = 'function/functional_scenetext_'

        if split == 'test':
            split='val'

        ann_file = None
        if SorF == 's':
            ann_file = sub_task_s[sub_data] + split + '.json'
        if SorF == 'f':
            ann_file = sub_task_f[sub_data] + split + '.json'

        path = '/root/CLOVE/json/'
        ann_file = path + ann_file

        self.split = split        
        self.ann = None
        
        self.ann = json.load(open(ann_file,'r'))

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        answer_list = '/root/CLOVE/json/answer_list.json'
        if split=='test' or split=='val':
            self.max_ques_words = 50 # do not limit question length during test
        self.answer_list = json.load(open(answer_list,'r'))    
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if ann['image_source']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image_id']+'.jpg')    
        elif ann['image_source']=='vg':
            image_path = os.path.join(self.vg_root,ann['image_id']+'.jpg')  
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test' or self.split=='val':
            question = pre_question(ann['question'],self.max_ques_words)   
            question_id = ann['question_id'] 
            answers = [ann['answers']][0]
            return image, question, question_id, answers


        elif self.split=='train':                       
            
            question = pre_question(ann['question'],self.max_ques_words)        
            
            if ann['image_source']=='vqa':
                
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answers'])
                    else:
                        answer_weight[answer] = 1/len(ann['answers'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['image_source']=='vg':
                answers = [ann['answers']][0]
                weights = [0.1] * 10  


            answers = [answer+self.eos for answer in answers]
                
            return image, question, answers, weights