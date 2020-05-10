import os
import json
from all_parameters import Params
import time
from datetime import datetime
#import request
from collections import defaultdict
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

from utils import string_factorlist2idxdixt_returner, from_KBdatasetdir_return_unique_entities_and_relations_dict, from_KBdatasetdir_return_idxnized_triplets_data
from utils import entity_symbol2canonical_and_entity_symbol2definitions___2___entity_symbol2cano_and_entity_symbol2def
class KBandDef_DatasetLoader(DatasetReader):
    def __init__(self, args, token_indexers=None):
        super().__init__(lazy=args.allen_lazyload)

        self.args = args
        self.min_padding_length = 0
        self.debug = self.args.debug
        self.use_reverse_rel_too = args.use_reverse_rel_too
        if self.args.embedding_strategy == 'elmo':
            self.max_seq_num = self.args.maxdeftoken_num_for_gpureserve_forelmo
        else:
            self.max_seq_num = self.args.maxdeftoken_num_for_gpureserve
        ############
        ### for bert only
        self.ifbert_use_whichmodel = args.ifbert_use_whichmodel # general, scibert, biobert
        self.bert_src_dir = args.bert_src_dir
        self.bert_word_first_last_all = args.bert_word_first_last_all

        self.maxdeftoken_num_for_gpureserve = self.args.maxdeftoken_num_for_gpureserve

        self.bertmodel_dir = ''
        if self.ifbert_use_whichmodel == 'general':
            self.bertmodel_dir += 'bert-base-uncased/' # recomendded ver is uncased, in original repository
            self.do_lower_case = True
            self.bertmodel_relative_dirpath = self.bert_src_dir + self.bertmodel_dir
            self.vocabpath = self.bertmodel_relative_dirpath + 'bert-base-uncased-vocab.txt'
            self.bertmodel_relative_dirpath = copy.copy('bert-base-uncased') # replace dir to model name
        elif self.ifbert_use_whichmodel == 'scibert':
            self.bertmodel_dir += 'scibert_scivocab_uncased/' # recomendded ver is uncased, in original repository
            self.do_lower_case = True
            self.bertmodel_relative_dirpath = self.bert_src_dir + self.bertmodel_dir
            self.vocabpath = self.bertmodel_relative_dirpath + 'vocab.txt'
        elif self.ifbert_use_whichmodel == 'biobert':
            self.bertmodel_dir += 'biobert_v1.1_pubmed/' # currently cased version only supported
            self.do_lower_case = False
            self.bertmodel_relative_dirpath = self.bert_src_dir + self.bertmodel_dir
            self.vocabpath = self.bertmodel_relative_dirpath + 'vocab.txt'

        ### for bert only end
        ##########

        ##########
        ### --- token indexer
        if args.embedding_strategy == 'pretrained':
            self.token_indexers = {'tokens':SingleIdTokenIndexer(token_min_padding_length=self.min_padding_length)}
        elif args.embedding_strategy == 'elmo':
            self.token_indexers = {'tokens':ELMoTokenCharactersIndexer(token_min_padding_length=self.min_padding_length)}
        elif args.embedding_strategy == 'bert':
            self.token_indexers = {'tokens':PretrainedBertIndexer(
                pretrained_model=self.mention_iterator.vocabpath,
                do_lowercase=self.mention_iterator.do_lower_case)
            }
            if args.ifbert_use_whichmodel in ['general', 'scibert']:
                self.do_lower_case = True
            elif args.ifbert_use_whichmodel == 'biobert':
                self.do_lower_case = False
        ### --- token indexer end
        ##########


        ### for specified KB and definitiondataset loading
        # Note If indexing finished, switch -cachedload True
        self.dataset = self.args.KBdataset
        self.specified_datasetdir = self.args.KBdataset_dir + self.dataset + '/'

        '''
        Required:
        Under ./data/dataset/ (example, ./data/FB15k/),  We need 8 files.
        
        train.txt, add_reverse_train.txt, valid.txt, add_reverse_valid.txt, test.txt, add_reverse_test.txt
        entity_symbol2cano_unkpad.json, entity_symbol2desc_unkpad.json
        
        Some datasets(e.g. DBPedia and WN18) don't have types, we currently won't construct entity_symbol2type or models using it. 
        '''
        ##########
        ### indexnized entity and relation
        if self.use_reverse_rel_too:
            cached_idxnized_dict_pklpath = self.specified_datasetdir +  'add_reverse_' +self.args.cached_idxnized_entity_symbol_and_relations
        else:
            cached_idxnized_dict_pklpath = self.specified_datasetdir + self.args.cached_idxnized_entity_symbol_and_relations
        print('creating/loading idxnized entity_symbols and relations dict...')
        self.uniq_entities_in_trn_valid_test, self.entity_symbol2idx, self.idx2entity_symbol, self.uniq_relations_in_trn_valid_test, self.relation2idx, self.idx2relation = from_KBdatasetdir_return_unique_entities_and_relations_dict(self.specified_datasetdir,
                                                                                                                                                                                                                                            cached_idxnized_dict_pklpath,
                                                                                                                                                                                                                                            os.path.exists(cached_idxnized_dict_pklpath),
                                                                                                                                                                                                                                            self.use_reverse_rel_too)
        print('idxnized entity_symbols and relations dict loaded!')
        ### indexnized entity and relations end
        ##########
        ##########
        ### indexnized triplets
        print('indexnized triplets start...')
        if self.use_reverse_rel_too:
            cached_indexnized_triplet_trn_valid_test_pklpath = self.specified_datasetdir + 'add_reverse_' + self.args.cached_idxnized_triplet_dataset_pklpath
        else:
            cached_indexnized_triplet_trn_valid_test_pklpath = self.specified_datasetdir + self.args.cached_idxnized_triplet_dataset_pklpath

        self.train_triplets, self.valid_triplets, self.test_triplets, self.all_triplets, self.train_triplets_idx, self.valid_triplets_idx, self.test_triplets_idx = from_KBdatasetdir_return_idxnized_triplets_data(KBdatasetdir_which_has_train_valid_test=self.specified_datasetdir,
                                                                                                                                                                                                                       cached_idxnized_dict_pklpath=cached_idxnized_dict_pklpath,
                                                                                                                                                                                                                       cached_indexnized_triplet_trn_valid_test_pklpath=cached_indexnized_triplet_trn_valid_test_pklpath,
                                                                                                                                                                                                                       pkl_file_exists=os.path.exists(
                                                                                                                                                                                                                           cached_indexnized_triplet_trn_valid_test_pklpath
                                                                                                                                                                                                                       ),
                                                                                                                                                                                                                       reverse_opt=self.use_reverse_rel_too)
        ### indexnized triplets end
        ##########
        cached_entity_symbol_idx2canonical_strings_and_entity_definitions_pklpath = self.specified_datasetdir + 'cached_entity_symbol_idx2canonical_strings_and_entity_definitions.pkl'
        self.entsymbolidx2cano, self.entsymbolidx2def = entity_symbol2canonical_and_entity_symbol2definitions___2___entity_symbol2cano_and_entity_symbol2def(cached_entity_symbol_idx2canonical_strings_and_entity_definitions_pklpath,
                                                                                                         self.entity_symbol2idx,
                                                                                                         self.specified_datasetdir)
        print('entsymbolidx2cano and def loaded!')

        self.dataset_instance_cached_dir = self.specified_datasetdir + 'Instance_cached/'
        self.dataset_instance_cached_dir_train_augmented = self.specified_datasetdir + 'trainaugmented_Instance_cached/'
        if not os.path.isdir(self.dataset_instance_cached_dir):
            os.mkdir(self.dataset_instance_cached_dir)
        if not os.path.isdir(self.dataset_instance_cached_dir_train_augmented):
            os.mkdir(self.dataset_instance_cached_dir_train_augmented)

        self.original_training_triplets_data_counts = len(self.train_triplets)
        self.whether_head_and_tail_has_no_sent_triplet = 0
        self.augumented_training_data_counts = 0
        self.all_datacounts = 0

        self.er_vocab = self.get_er_vocab()

        if self.debug:
            self.train_triplets, self.valid_triplets, self.test_triplets, self.all_triplets, self.train_triplets_idx, self.valid_triplets_idx, self.test_triplets_idx = self.debugdataloader(
                self.train_triplets, self.valid_triplets, self.test_triplets, self.all_triplets,
                self.train_triplets_idx, self.valid_triplets_idx, self.test_triplets_idx)


    def dataset_statistics_returner(self):
        print('\n######## DATASET STATISTICS')
        print('original training data', self.original_training_triplets_data_counts)
        print('augmented training data', self.augumented_training_data_counts)
        print('valid data', len(self.valid_triplets))
        print('test data', len(self.test_triplets))
        print('all data ignoring augmented training data(original train+valid+test)', len(self.all_triplets))
        print('augment sent by canonical triplets over train, valid, test(aug train included)',
              self.whether_head_and_tail_has_no_sent_triplet,'/',
              self.augumented_training_data_counts + len(self.valid_triplets) + len(self.test_triplets))

    def simply_return_head_rel_tail_one_sent_of_head_and_tail(self, triplets):
        '''
        :param triplets: [3, 4, 1]
        :return: {'head': 3, 'rel':4, 'tail':1, 'head_sent':['Only', 'one', 'sentence'], 'tail_sent': ['Only', 'one', 'sentence'],
                 'head_cano': ['DNA', '33'], 'tail_cano': ['ATP','325']}
        '''
        # NOTE: Even if entity has multi sentences, choice only first sentence
        # This function is mainly for valid and test data.
        #pdb.set_trace()
        head_sent = self.entsymbolidx2def[triplets[0]][0]
        tail_sent = self.entsymbolidx2def[triplets[2]][0]
        head_cano = self.entsymbolidx2cano[triplets[0]]
        tail_cano = self.entsymbolidx2cano[triplets[2]]

        whether_head_and_tail_has_no_sent_triplet_flag = 0
        if head_sent == ['@@UNKNOWN@@']:
            head_sent = copy.copy(head_cano)
            whether_head_and_tail_has_no_sent_triplet_flag += 1
        if tail_sent == ['@@UNKNOWN@@']:
            tail_sent = copy.copy(tail_cano)
            whether_head_and_tail_has_no_sent_triplet_flag += 1
        if whether_head_and_tail_has_no_sent_triplet_flag:
            self.whether_head_and_tail_has_no_sent_triplet += 1
        return {'head': triplets[0],
                'rel': triplets[1],
                'tail': triplets[2],
                'head_sent': head_sent,
                'tail_sent': tail_sent,
                'head_cano': head_cano,
                'tail_cano': tail_cano
                }

    def augument_train_by_multisent(self,triplets):
        '''
        :param triplets: indexnized, ex. [3, 4, 1]
        :return: list: [{'head': 3, 'rel':4, 'tail':1, 'head_sent':[...], 'tail_sent': [...]}]
        '''
        head_sent_unkcheck = self.entsymbolidx2def[triplets[0]][0]
        tail_sent_unkcheck = self.entsymbolidx2def[triplets[2]][0]
        head_cano = self.entsymbolidx2cano[triplets[0]]
        tail_cano = self.entsymbolidx2cano[triplets[2]]
        head_sent = self.entsymbolidx2def[triplets[0]]
        tail_sent = self.entsymbolidx2def[triplets[2]]
        whether_head_and_tail_has_no_sent_triplet_flag = 0
        if head_sent_unkcheck == ['@@UNKNOWN@@']:
            head_sent = [copy.copy(head_cano)]
            whether_head_and_tail_has_no_sent_triplet_flag += 1
        if tail_sent_unkcheck == ['@@UNKNOWN@@']:
            tail_sent = [copy.copy(tail_cano)]
            whether_head_and_tail_has_no_sent_triplet_flag += 1
        if whether_head_and_tail_has_no_sent_triplet_flag:
            self.whether_head_and_tail_has_no_sent_triplet += 1

        argumented_data_list = []

        for one_head_sent in head_sent:
            for one_tail_sent in tail_sent:
                argumented_data_list.append({'head': triplets[0],
                'rel': triplets[1],
                'tail': triplets[2],
                'head_sent': one_head_sent,
                'tail_sent': one_tail_sent,
                'head_cano': head_cano,
                'tail_cano': tail_cano})

        return argumented_data_list

    @overrides
    def text_to_instance(self,data=None, train_valid_test_flag='train') -> Instance:
        '''
        :param data:
               {'head': triplets[0],    int
                'rel': triplets[1],     int
                'tail': triplets[2],    int
                'head_sent': head_sent, tokenized_tokenlist
                'tail_sent': tail_sent, tokenized_tokenlist
                'head_cano': head_cano, tokenized_tokenlist
                'tail_cano': tail_cano  tokenized_tokenlist
                }
        :return:
        '''
        fields = {}
        fields['head'] = ArrayField(np.array([data['head']], dtype='int32'))
        fields['relation'] = ArrayField(np.array([data['rel']], dtype='int32'))
        fields['tail'] = ArrayField(np.array([data['tail']], dtype='int32'))
        fields['head_ent_sent'] = TextField([Token(word) for word in data['head_sent']][:self.max_seq_num ], self.token_indexers)
        fields['tail_ent_sent'] = TextField([Token(word) for word in data['tail_sent']][:self.max_seq_num], self.token_indexers)
        fields['head_sentence_length'] = ArrayField(np.array([len([Token(word) for word in data['head_sent']][:self.max_seq_num])], dtype='int32'))
        fields['tail_sentence_length'] = ArrayField(np.array([len([Token(word) for word in data['tail_sent']][:self.max_seq_num])], dtype='int32'))
        fields['head_cano'] = TextField([Token(word) for word in data['head_cano']][:self.max_seq_num], self.token_indexers)
        fields['tail_cano'] = TextField([Token(word) for word in data['tail_cano']][:self.max_seq_num], self.token_indexers)
        #pdb.set_trace()
        if train_valid_test_flag == 'train' or train_valid_test_flag == 'valid':
            fields['tail_golds'] = ArrayField(np.array(self.er_vocab[(data['head'], data['rel'])][:self.args.batch_size], dtype='int32'), padding_value=-1)
        else:
            fields['tail_golds'] = ArrayField(np.array(self.er_vocab[(data['head'], data['rel'])], dtype='int32'), padding_value=-1)
        #pdb.set_trace()


        return Instance(fields)

    def invalid_data_flagchecker(self,data):
        flag = 0
        if self.er_vocab[(data['head'], data['rel'])] == []:
            flag += 1
        if len(data['head_sent']) == 0:
            flag += 1
        if len(data['tail_sent']) == 0:
            flag += 1
        if len(data['head_cano']) == 0:
            flag += 1
        if len(data['tail_cano']) == 0:
            flag += 1
        if flag:
            print('\n *****INVALID DATA:' ,data, '*****\n')
        return flag

    @overrides
    def _read(self, train_or_valid_or_test_flag) -> Iterator[Instance]:
        if train_or_valid_or_test_flag == 'train':
            for idx, triplet in tqdm(self.train_triplets.items()):
                self.all_datacounts += 1
                if self.args.training_dataset_augument_by_multisent:
                    augument_triplets_data_list = self.augument_train_by_multisent(triplet)
                    for triplet_datas in augument_triplets_data_list:
                        self.augumented_training_data_counts += 1
                        invalidflag = self.invalid_data_flagchecker(triplet_datas)
                        if not invalidflag:
                            yield self.text_to_instance(data=triplet_datas,train_valid_test_flag=train_or_valid_or_test_flag)
                else:
                    triplet_datas = self.simply_return_head_rel_tail_one_sent_of_head_and_tail(triplet)
                    invalidflag = self.invalid_data_flagchecker(triplet_datas)
                    if not invalidflag:
                        yield self.text_to_instance(data=triplet_datas,train_valid_test_flag=train_or_valid_or_test_flag)
        elif train_or_valid_or_test_flag == 'valid':
            for idx, triplet in tqdm(self.valid_triplets.items()):
                self.all_datacounts += 1
                triplet_datas = self.simply_return_head_rel_tail_one_sent_of_head_and_tail(triplet)
                invalidflag = self.invalid_data_flagchecker(triplet_datas)
                if not invalidflag:
                    yield self.text_to_instance(data=triplet_datas,train_valid_test_flag=train_or_valid_or_test_flag)
        elif train_or_valid_or_test_flag == 'test':
            for idx, triplet in tqdm(self.test_triplets.items()):
                self.all_datacounts += 1
                triplet_datas = self.simply_return_head_rel_tail_one_sent_of_head_and_tail(triplet)

                invalidflag = self.invalid_data_flagchecker(triplet_datas)
                if not invalidflag:
                    yield self.text_to_instance(data=triplet_datas,train_valid_test_flag=train_or_valid_or_test_flag)
        else:
            print('invalid data flag. choose train/valid/test')

    ### miscs(needed)
    def get_er_vocab(self):
        self.get_er_vocab_path = self.specified_datasetdir + 'er_vocab.pickle'
        if not os.path.isdir(self.get_er_vocab_path):
            er_vocab = defaultdict(list)
            for data in [self.train_triplets, self.valid_triplets, self.test_triplets]:
                for idx, triplet in data.items():
                    er_vocab[(triplet[0], triplet[1])].append(triplet[2])
            with open(self.get_er_vocab_path, 'wb') as gv:
                pickle.dump(er_vocab,gv)
            return er_vocab
        else:
            with open(self.get_er_vocab_path, 'rb') as gvl:
                return pickle.load(gvl)


    def debugdataloader(self, train_triplets, valid_triplets, test_triplets, all_triplets, train_triplets_idx,
                        valid_triplets_idx, test_triplets_idx):
        print('\n#### DEBUG ON ####\n')
        debug_datanum = self.args.debug_datapoints_for_train_valid_test
        dtrain, dvalid, dtest, dall, dtrainidx, dvalididx, dtestidx = {}, {}, {}, {}, [], [], []
        for idx, (dataidx, itsdata) in enumerate(train_triplets.items()):
            dtrain.update({dataidx: itsdata})
            dall.update({dataidx: itsdata})
            dtrainidx.append(dataidx)
            if idx == debug_datanum:
                break
        for idx, (dataidx, itsdata) in enumerate(valid_triplets.items()):
            dvalid.update({dataidx: itsdata})
            dall.update({dataidx: itsdata})
            dvalididx.append(dataidx)
            if idx == debug_datanum // 2:
                break
        for idx, (dataidx, itsdata) in enumerate(test_triplets.items()):
            dtest.update({dataidx: itsdata})
            dall.update({dataidx: itsdata})
            dtestidx.append(dataidx)
            if idx == debug_datanum // 2:
                break
        return dtrain, dvalid, dtest, dall, dtrainidx, dvalididx, dtestidx

    def get_not_augmented_cached_dir(self):
        return self.dataset_instance_cached_dir

    def get_augmented_cached_dir(self):
        return self.dataset_instance_cached_dir_train_augmented

    def get_relation_num(self):
        return len(self.relation2idx)

    def get_entities_num(self):
        return len(self.entity_symbol2idx)




