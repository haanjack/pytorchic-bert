# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn

import checkpoint
import optim

try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this.")


class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    loss_scale: int = 0 # loss_scale for mixed precision training

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, dataloader, lr, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.dataloader = dataloader # data loader
        self.learning_rate = lr
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, get_loss, model_file=None, pretrain_file=None, num_gpu=1, fp16=False, local_rank=-1):
        """ Train Loop """
        self.model.train() # train mode
        self.load(model_file, pretrain_file)

        if fp16:
            self.model = self.model.half()
        model = self.model.to(self.device)
        if local_rank != -1: # use Distributed Data Parallelism for multi-node training
            model = DDP(model)
        elif num_gpu > 1: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        # placed optimizer after the model preparation
        optimizer = optim.optim4GPU(self.cfg, self.model, fp16)

        ## dbug
        for group in optimizer.param_groups:
            for p in group['params']:
                print(p.device)
                break

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.dataloader, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                loss = get_loss(model, batch, global_step)
                if num_gpu > 1:
                    loss = loss.mean() # mean() for Data Parallelism

                if fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                loss_sum += loss.item()
                # modify learning rate with special warm up BERT uses
                lr_this_step = self.learning_rate * optim.warmup_linear(global_step/self.cfg.total_steps, self.cfg.warmup)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, num_gpu=1):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        # TODO: distributed???
        if num_gpu > 1: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        iter_bar = tqdm(self.dataloader, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                accuracy, result = evaluate(model, batch) # accuracy to print
            results.append(result)

            iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        return results

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts


    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

