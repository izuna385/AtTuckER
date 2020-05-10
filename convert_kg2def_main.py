import os
import json
from all_parameters import Params
import time
from datetime import datetime
#import request
import numpy as np
import gc
import copy
from tqdm import tqdm
import adabound
import torch
from torch.nn import LSTM, ModuleList
import torch.optim as optim
from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, BagOfEmbeddingsEncoder
import torch.nn as nn
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
# from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
# NOTE: we have to load BasicTextFieldEmbedder and PretrainedLoader under same .py file
# (Don't know the reason...)
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
# We have to convert word to Token instances
from allennlp.data.tokenizers import Token
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
import glob
import pickle
import argparse
import logging
import joblib
import sys
import pdb
from overrides import overrides
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate
from torch.optim import SGD, Optimizer
from tqdm import tqdm
from pytorch_transformers import *
from allennlp.data.iterators import BasicIterator
from allennlp.common.tee_logger import TeeLogger # Logging with tee
from pytz import timezone
import random
from distutils.util import strtobool
from memory_profiler import profile
from multiprocessing import Pool
import multiprocessing as multi
from utils import string_factorlist2idxdixt_returner, from_KBdatasetdir_return_unique_entities_and_relations_dict, datasetteelogger
from dataloader import KBandDef_DatasetLoader
from word_embed import Embeddings_caller
from sentence_encoder import RelationAttendedDefinitionSentenceEncoder, DefinitionSentenceEncoder
from model import InBatchDefinitionEncTucker
from evaluate_all_ent import EvaluatorClass
from convert_kg2def_model import DefinitionVAE, SimpleKGProjector
torch.backends.cudnn.deterministic = True
seed = 777
np.random.seed(seed)
torch.manual_seed(seed)

from convert_kg2def_dataloader import KG2def_dataloader, KG2def_datasetReader, ProjectedKGembReader

def main():
    params_class = Params()
    opts = params_class.incomplete_def_kgemb2defemb_converter()
    train_datasetReader = KG2def_datasetReader(args=opts)
    experiment_dir = train_datasetReader.specified_datasetdir
    if 1: #opts.teelog and opts.debug == False:
        dir_for_each_experiment = datasetteelogger(experiment_dir)

    train_data_all = train_datasetReader.from_entity_symbol_2kgemb_end_defs_dumppath_dataloader()
    train_data_aug_or_not = train_datasetReader.from_all_data_onesentence_extraction_or_augmentation_data_creation(opts, train_data_all)

    trains = train_datasetReader.read(train_data_aug_or_not)

    vocab = Vocabulary.from_instances(trains) if opts.embedding_strategy == 'pretrained' else  Vocabulary()
    iterator = BucketIterator(batch_size=opts.batch_size,sorting_keys=[('target_definitions','num_tokens')])
    iterator.index_with(vocab)
    emb_mapper, emb_dim, textfieldEmbedder = Embeddings_caller(opts).embeddings_returner(vocab=vocab)

    encoder = DefinitionSentenceEncoder(args=opts, input_dim=emb_dim, hidden_dim=emb_dim, word_embedder=textfieldEmbedder)

    # Todo: target implementation
    # model = DefinitionVAE(args=opts, kgemb_dim=300, encoder=encoder, vocab=vocab)

    model = SimpleKGProjector(args=opts, kgemb_dim=300, encoder=encoder, vocab=vocab)

    if torch.cuda.is_available():
        model = model.cuda()

    if opts.optimizer == 'AdaBound':
        optimizer = adabound.AdaBound(filter(lambda param: param.requires_grad, model.parameters()), lr=opts.lr,
                                      eps=opts.epsilon,weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), final_lr=0.1)
    elif opts.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=opts.lr,
                               eps=opts.epsilon,weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), amsgrad=opts.amsgrad)
    elif opts.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr=opts.lr,
                               eps=opts.epsilon,weight_decay=opts.weight_decay,amsgrad=opts.amsgrad)

    else:
        assert opts.optimizer == 'SGD'
        optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=opts.lr,weight_decay=opts.weight_decay,momentum=opts.momentum)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=trains,
                      # grad_clipping=opts.grad_clipping,
                      cuda_device=[int(i) for i in opts.cuda_device.split(',')],
                      num_epochs=opts.num_epochs
                      )
    trainer.train()

    linear_weight_data = model.linear_weight_dumper() # tensor
    weight_path = dir_for_each_experiment + 'linear_weight.pkl'
    with open(weight_path, 'wb') as wp:
        pickle.dump(linear_weight_data, wp)
    print('linear weight dumped to',weight_path)
    # entity_symbol2emb_dict = train_datasetReader.tsvparser()
    # forvae_Datareader = ProjectedKGembReader(args=opts, linear_weight=linear_weight_data, entity_symbol2emb_dict=entity_symbol2emb_dict)
    # trains = forvae_Datareader.read(entity_symbol2emb_dict)

if __name__ == '__main__':
    main()