import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question


class clove_dataset(Dataset):
    def __init__(self, ann_file, transform, root, eos='[SEP]', split="train", max_ques_words=30, answer_list='', SorF='s', sub_data='a'):
        sub_task_s = dict()
        sub_task_s['a'] = 'json/scene/scene_a#ShopAndDining_'
        sub_task_s['b'] = 'json/scene/scene_b#Workplace_'
        sub_task_s['c'] = 'json/scene/scene_c#HomeOrHotel_'
        sub_task_s['d'] = 'json/scene/scene_d#Transportation_'
        sub_task_s['e'] = 'json/scene/scene_e#SportAndLeisure_'
        sub_task_s['f'] = 'json/scene/scene_f#Outdoors_'

        sub_task_f = dict()
        sub_task_f['a'] = 'json/function/functional_attribute_'
        sub_task_f['b'] = 'json/function/functional_knowledge_'
        sub_task_f['c'] = 'json/function/functional_logical_'
        sub_task_f['d'] = 'json/function/functional_object_'
        sub_task_f['e'] = 'json/function/functional_relation_'
        sub_task_f['f'] = 'json/function/functional_scenetext_'

        if split == 'test':
            split='val'

        ann_file = None
        if SorF == 's':
            ann_file = sub_task_s[sub_data] + split + '.json'
        if SorF == 'f':
            ann_file = sub_task_f[sub_data] + split + '.json'

        #path = '/project/rostamim_919/caiyulia/data/CLOVE/'
        path = root
        ann_file = path + ann_file

        self.split = split        
        self.ann = None
        
        self.ann = json.load(open(ann_file,'r'))

        self.transform = transform
        self.vg_root = path + 'vg/image'
        self.textvqa_root = path + 'textvqa'
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        answer_list = path + 'answer_list.json'
        if split=='test' or split=='val':
            self.max_ques_words = 50 # do not limit question length during test
        self.answer_list = json.load(open(answer_list,'r'))    
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if ann['image_source']=='vg':
            image_path = os.path.join(self.vg_root,ann['image_id']+'.jpg')  
        elif ann['image_source']=='textvqa':
            image_path = os.path.join(self.textvqa_root,ann['image_id']+'.jpg')  
            
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

            elif ann['image_source']=='vg' or ann['image_source']=='textvqa':
                answers = [ann['answers']][0]
                weights = [0.1] * 10  


            answers = [answer+self.eos for answer in answers]
                
            return image, question, answers, weights