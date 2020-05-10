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

torch.backends.cudnn.deterministic = True
seed = 777 # Don't change this seeds.
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    params_class = Params()
    opts = params_class.get_params_for_model_training()
    KBandRelateddataReader = KBandDef_DatasetLoader(args=opts)
    KBspecified_experiment_dir = KBandRelateddataReader.specified_datasetdir
    if opts.teelog and opts.debug == False:
        dir_for_each_experiment = datasetteelogger(KBspecified_experiment_dir)
    trains, valids, tests = KBandRelateddataReader.read('train'), KBandRelateddataReader.read('valid'), KBandRelateddataReader.read('test')
    KBandRelateddataReader.dataset_statistics_returner()

    # Call vocabs and embeddings
    vocab = Vocabulary.from_instances(trains + valids + tests) if opts.embedding_strategy == 'pretrained' else  Vocabulary()
    iterator = BucketIterator(batch_size=opts.batch_size,sorting_keys=[('head_ent_sent','num_tokens')])
    iterator.index_with(vocab)
    emb_mapper, emb_dim, textfieldEmbedder = Embeddings_caller(opts).embeddings_returner(vocab=vocab)
    er_vocab = KBandRelateddataReader.get_er_vocab()

    # NOTE: This has no means?
    if opts.relation_attention2headandtaildef:
        encoder = RelationAttendedDefinitionSentenceEncoder(args=opts, input_dim=emb_dim, hidden_dim=emb_dim ,word_embedder=textfieldEmbedder)
    else:
        encoder = DefinitionSentenceEncoder(args=opts, input_dim=emb_dim, hidden_dim=emb_dim ,word_embedder=textfieldEmbedder)

    model = InBatchDefinitionEncTucker(args=opts,word_embeddings=textfieldEmbedder,encoder=encoder,relation_nums=KBandRelateddataReader.get_relation_num(),d1=opts.ent_dim,d2=opts.rel_dim,
                                       vocab=vocab,er_vocab=er_vocab)
    model.init()
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
                      grad_clipping=opts.grad_clipping,
                      validation_dataset=tests,
                      cuda_device=[int(i) for i in opts.cuda_device.split(',')],
                      num_epochs=opts.num_epochs
                      )
    print('\n####\nTRAINING START\n####')
    trainer.train()
    print('####### TRAINING FINISHED #######')

    # Prepare for Evaluation
    all_entity_num = KBandRelateddataReader.get_entities_num()
    Evaluator = EvaluatorClass(args=opts, model=model, vocab=vocab, er_vocab=model.get_er_vocab(), all_entity_num=all_entity_num, entity_dim=opts.ent_dim)

    # First, construct entity_symbolidx2_encoded_sentvector_projected_to_KBentdim
    # This matrix will also use for entity linking.
    all_ent_Emb = Evaluator.get_E_numpy_from_alldataset(dslist=[trains, valids, tests])
    '''
    get_E_numpy_from_alldataset only converts entitiy's definitions to vector.
    Since the assumption is that we can see all entities' definitions, this act is no problem.
    (Also, this assumption can be applied to open settings
    '''
    Evaluator.add_Embclass_2_model(E_numpy_weight=all_ent_Emb)
    Evaluator.evaluation(ds=tests)

if __name__ == '__main__':
    main()