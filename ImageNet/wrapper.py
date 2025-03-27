# Copyright 2024 Lawrence Livermore National Security, LLC and other
# Authors: Vivek Sivaraman Narayanaswamy. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import numpy as np

class AnchoringWrapper(nn.Module):
    def __init__(self,base_network,crop=0,imsize=224,flip=False):
        super(AnchoringWrapper,self).__init__()
        '''

        base_network (default: None):
            network used to perform anchor training (takes 6 input channels)

        '''            
        self.net = base_network
        self.crop = crop
        self.flip = flip
        self.imsize = imsize
        self.mask = transforms.Compose([
                        transforms.CenterCrop(self.crop),
                        transforms.Pad((self.imsize-self.crop)//2)
                    ])

    def masking(self,samples):
        return self.mask(samples)

    def create_anchored_batch(self,x,anchors=None,n_anchors=1,corrupt=False):
        '''
        anchors (default=None):
            if passed, will use the same set of anchors for all batches during training.
            if  None, we will use a shuffled input minibatch to forward( ) as anchors for that batch (random)

            During *inference* it is recommended to keep the anchors fixed for all samples.

            n_anchors is chosen as min(n_batch,n_anchors)
        '''
        
        n_img = x.shape[0]
        if anchors is None:
            anchors = x[torch.randperm(n_img),:]
        
        ## make anchors (n_anchors) --> n_img*n_anchors
        if self.training:
            A = anchors[torch.randint(anchors.shape[0],(n_img*n_anchors,)),:]
        else:
            A = torch.repeat_interleave(anchors[torch.randperm(n_anchors),:],n_img,dim=0) 

        if corrupt:
            refs = self.masking(A)
        else:
            refs = A

        ## before computing residual, make minibatch (n_img) --> n_img* n_anchors

        if len(x.shape)<=2:

            diff = x.tile((n_anchors,1))
            assert diff.shape[1]==A.shape[1], f"Tensor sizes for `diff`({diff.shape}) and `anchors` ({A.shape}) don't match!"
            diff -= A
        else:
            diff = x.tile((n_anchors,1,1,1)) - A
        
        batch = torch.cat([refs,diff],axis=1)

        return batch

    def calibrate(self,mu,sig):
        c = torch.mean(sig,1)
        c = c.unsqueeze(1).expand(mu.shape)
        return torch.div(mu,1+torch.exp(c))
        # return torch.div(mu,c)

    def forward_original(self,x,anchors=None,corrupt=False,n_anchors=1,return_std=False):
        if n_anchors==1 and return_std:
            raise Warning('Use n_anchor>1, std. dev cannot be computed!')

        a_batch = self.create_anchored_batch(x,anchors=anchors,n_anchors=n_anchors,corrupt=corrupt)

        p = self.net(a_batch)

        if return_std:
            p = p.reshape(n_anchors,x.shape[0],p.shape[1])
            mu = p.mean(0)
            std = p.sigmoid().std(0)
            return self.calibrate(mu,std), std
        elif n_anchors>1:
            p = p.reshape(n_anchors,x.shape[0],p.shape[1])
            mu = p.mean(0)
            std = p.sigmoid().std(0)
            return self.calibrate(mu,std)
        else:
            return p
    
    def forward(self,x,n_anchors=1,anchors=None,corrupt=False):
       return self.forward_original(x,n_anchors=n_anchors,anchors=anchors,corrupt=corrupt)
        
        
class AnchoringWrapperInference(AnchoringWrapper):
    def __init__(self,base_network):
        super(AnchoringWrapper, self).__init__(base_network)
        '''

        base_network (default: None):
            network used to perform anchor training (takes 6 input channels)

        '''            
        self.net = base_network
    
    def forward(self,x,anchors,n_anchors=1,return_std=False):
        return self.forward_original(x,n_anchors=n_anchors,anchors=anchors,return_std=return_std)

          