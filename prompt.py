import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist


class DualKeyPrompt_cluster(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='cls', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, image_key_size=False, text_key_size=False, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', general_prompt_length=10):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.image_key_size = image_key_size
        self.text_key_size = text_key_size
        self.pool_size = image_key_size * text_key_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.freq_list = torch.zeros(self.pool_size)
        # test adding general prompt
        self.general_prompt = nn.Parameter(torch.randn(general_prompt_length,embed_dim)).to('cuda')
        if self.prompt_pool:
            prompt_pool_shape = (self.pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape)).to('cuda')
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)).to('cuda')
                nn.init.uniform_(self.prompt, -1, 1)
        self.max_change_size = 1000
        self.max_change_sim = 0.5
        # if using learnable prompt keys
        if prompt_key:
            text_key_shape = (self.text_key_size, embed_dim)
            image_key_shape = (self.image_key_size, embed_dim)
            if prompt_key_init == 'zero':
                self.text_prompt_key = nn.Parameter(torch.zeros(text_key_shape))
                self.img_prompt_key = nn.Parameter(torch.zeros(image_key_shape))
            elif prompt_key_init == 'uniform':
                self.text_prompt_key = torch.randn(text_key_shape).to('cuda')
                self.img_prompt_key = torch.randn(image_key_shape).to('cuda')
                nn.init.uniform_(self.text_prompt_key, -1, 1)
                nn.init.uniform_(self.img_prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, text_embed, img_embed, prompt_mask=None, cls_features=None, word_idx = None, add_prompt = False):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                text_embed_mean = torch.mean(text_embed, dim=1)
                img_embed_mean = torch.mean(img_embed, dim=1)
            elif self.embedding_key == 'max':
                text_embed_mean = torch.max(text_embed, dim=1)[0]
                img_embed_mean = torch.max(img_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                text_embed_mean = torch.max(text_embed, dim=1)[0] + 2 * torch.mean(text_embed, dim=1)
                img_embed_mean = torch.max(img_embed, dim=1)[0] + 2 * torch.mean(img_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    text_embed_mean = torch.max(text_embed, dim=1)[0] # B, C
                    img_embed_mean = torch.max(img_embed, dim=1)[0] # B, C
                else:
                    text_embed_mean = cls_features
                    img_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            text_prompt_norm = self.l2_normalize(self.text_prompt_key, dim=1) # Pool_size, C
            img_prompt_norm = self.l2_normalize(self.img_prompt_key, dim=1)
            text_embed_norm = self.l2_normalize(text_embed_mean, dim=1) # B, C
            img_embed_norm = self.l2_normalize(img_embed_mean, dim=1) # B, C
            text_similarity = torch.matmul(text_embed_norm, text_prompt_norm.t()) # B, Pool_size
            img_similarity = torch.matmul(img_embed_norm, img_prompt_norm.t()) # B, Pool_size

            '''text_prompt_norm = self.text_prompt_key
            img_prompt_norm = self.img_prompt_key
            text_embed_norm = text_embed_mean
            img_embed_norm = img_embed_mean

            # text 
            diff = text_embed_norm.unsqueeze(1) - text_prompt_norm.unsqueeze(0)
            text_similarity = torch.norm(diff, p=2, dim=2)
            # image
            diff = img_embed_norm.unsqueeze(1) - img_prompt_norm.unsqueeze(0)
            img_similarity = torch.norm(diff, p=2, dim=2)'''
            
            #similarity = text_similarity + img_similarity
            
            if prompt_mask is None:
                _, text_idx = torch.topk(text_similarity, k=self.top_k, dim=1) # B, top_k
                _, image_idx = torch.topk(img_similarity, k=self.top_k, dim=1) # B, top_k
                if self.top_k == 1:
                    idx = text_idx * self.text_key_size + image_idx
                    #idx = image_idx * self.image_key_size + text_idx
                else:
                    idx1 = text_idx[:,0] * self.text_key_size + image_idx[:,0]
                    idx2 = text_idx[:,1] * self.text_key_size + image_idx[:,0]
                    idx3 = text_idx[:,0] * self.text_key_size + image_idx[:,1]
                    idx = torch.stack([idx1,idx2,idx3],dim=-1)


            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C
            batched_general = self.general_prompt.unsqueeze(0)
            batched_general = batched_general.repeat(batch_size,1,1)
            batched_prompt = torch.cat([batched_general,batched_prompt],dim=1)
            
            out['prompt_idx'] = idx
            out['image_idx'] = image_idx
            out['text_idx'] = text_idx
            #for i in idx:
            #    self.freq_list[int(idx[i][0].item())] += 1
            #print(self.freq_list)

            # Debugging, return sim as well
            out['text_prompt_norm'] = text_prompt_norm
            out['img_prompt_norm'] = img_prompt_norm
            out['text_embed_norm'] = text_embed_norm
            out['img_embed_norm'] = img_embed_norm
            #out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_text_key_norm = text_prompt_norm[text_idx] # B, top_k, C
            out['selected_text_key'] = batched_text_key_norm
            batched_img_key_norm = img_prompt_norm[image_idx] # B, top_k, C
            out['selected_img_key'] = batched_img_key_norm
            text_embed_norm = text_embed_norm.unsqueeze(1) # B, 1, C
            img_embed_norm = img_embed_norm.unsqueeze(1)
            sim = batched_text_key_norm * text_embed_norm + batched_img_key_norm * img_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / text_embed.shape[0] # Scalar
            '''
            batched_text_key_norm_comp = text_prompt_norm[idx_complement]
            batched_img_key_norm_comp = img_prompt_norm[idx_complement]
            diff = text_embed_norm * batched_text_key_norm_comp + img_embed_norm * batched_img_key_norm_comp
            reduce_diff = (torch.sum(diff) / text_embed.shape[0]) / 10'''
            out['reduce_diff'] = 0

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['text_prompted_embedding'] = torch.cat([batched_prompt, text_embed], dim=1)
        out['img_prompted_embedding'] = torch.cat([batched_prompt, img_embed], dim=1)
        out['batched_prompt'] = batched_prompt
        out['modified'] = False


        return out
