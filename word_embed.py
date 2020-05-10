import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, pickle
import gc
import pdb
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.common import Params
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from overrides import overrides
from torch.autograd import Variable
import copy
from tqdm import tqdm
from scipy.special import expit
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
from torch.nn.functional import normalize
from allennlp.nn.util import get_text_field_mask, add_positional_features # TODO: where to add. Check with MultiheadAttention

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.init import xavier_normal, xavier_uniform
from torch.nn.functional import sigmoid

class Embeddings_caller(Model):
    def __init__(self, args):
        self.embedding_strategy = args.embedding_strategy # elmo or bert?
        self.args = args

        # requires_grad for elmo/bert embedding. Default:False
        self.emb_requires_grad = bool(self.args.emb_requires_grad)

        # bert config
        self.ifbert_use_whichmodel = args.ifbert_use_whichmodel # general, scibert, biobert

        self.bert_src_dir = args.bert_src_dir
        self.bert_word_first_last_all = args.bert_word_first_last_all
        self.bert_top_layer_only = args.bert_top_layer_only

        self.elmo_src_dir = args.elmo_src_dir
        self.ifelmo_use_whichmodel = args.ifelmo_use_whichmodel

        self.pretrained_src_dir = args.pretrained_src_dir

        if args.ifpretrained_use_whichmodel == 'glove':
            self.glove_embeddings_file = self.pretrained_src_dir + 'glove.840B.300d.txt'
        elif args.ifpretrained_use_whichmodel == 'bioglove':
            self.glove_embeddings_file = self.pretrained_src_dir + 'bio_GloVe_300.txt'
        elif args.ifpretrained_use_whichmodel == 'ext_biow2v':
            self.glove_embeddings_file = self.pretrained_src_dir + 'bio_emb_extrinsic.txt'

    def embeddings_returner(self, vocab=None):
        '''
        Either the name of the pretrained model to use (e.g. bert-base-uncased),or the path to the .tar.gz
        file with the model weights.
        :param args: vocab_size and vocab is needed only when pretrained embeddings is used.
        :return: embedder
        '''

        '''
        "bert-base-uncased", do_lower_case=True
        "bert-base-cased" , do_lower_case=False
        https://github.com/huggingface/pytorch-transformers/issues/712
        https://qiita.com/uedake722/items/b7f4b75b4d77d9bd358b
        '''
        if self.embedding_strategy == 'bert':
            self.bertmodel_dir = ''
            if self.ifbert_use_whichmodel == 'general':
                self.bertmodel_dir += 'bert-base-uncased/' # recomendded ver is uncased, in original repository
                self.bertmodel_relative_dirpath = self.bert_src_dir + self.bertmodel_dir

                # included in pytorch_transformers, so we replace it with model name itself
                self.bert_weight_filepath = copy.copy('bert-base-uncased')

            elif self.ifbert_use_whichmodel == 'scibert':
                self.bertmodel_dir += 'scibert_scivocab_uncased/' # recomendded ver is uncased, in original repository
                self.bertmodel_relative_dirpath = self.bert_src_dir + self.bertmodel_dir
                self.bert_weight_filepath = self.bertmodel_relative_dirpath + 'weights.tar.gz'

            elif self.ifbert_use_whichmodel == 'biobert':
                self.bertmodel_dir += 'biobert_v1.1_pubmed/' # currently cased version only supported
                self.bertmodel_relative_dirpath = self.bert_src_dir + self.bertmodel_dir
                self.bert_weight_filepath = self.bertmodel_relative_dirpath + 'weights.tar.gz' # including bert_config.json and bin.

            # Load embedder
            bert_embedder = PretrainedBertEmbedder(pretrained_model=self.bert_weight_filepath,
                                                   top_layer_only=self.bert_top_layer_only,
                                                   requires_grad=self.emb_requires_grad)
            return bert_embedder, bert_embedder.get_output_dim(), BasicTextFieldEmbedder({'tokens':bert_embedder},allow_unmatched_keys=True)

        elif self.embedding_strategy == 'elmo':
            if self.ifelmo_use_whichmodel == 'general':
                options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
                weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
            elif self.ifelmo_use_whichmodel == 'pubmed':
                options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json'
                weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5'
            elif self.ifelmo_use_whichmodel == 'bioelmo':
                options_file = self.elmo_src_dir + 'BioELMo/weights/biomed_elmo_options.json'
                weight_file = self.elmo_src_dir + 'BioELMo/weights/biomed_elmo_weights.hdf5'
            else:
                options_file = -1
                weight_file = -1
            assert options_file != -1
            elmo_embedder = ElmoTokenEmbedder(options_file=options_file, weight_file=weight_file,
                                              requires_grad=self.emb_requires_grad)
            return elmo_embedder ,elmo_embedder.get_output_dim(), BasicTextFieldEmbedder({'tokens':elmo_embedder})

        elif self.embedding_strategy == 'pretrained':

            print('\nGloVe pretrained vocab loading\n')

            if 'glove' in self.args.ifpretrained_use_whichmodel:
                embedding_dim = 300
            else:
                embedding_dim = 200

            pretrain_emb_embedder =  Embedding.from_params(vocab=vocab,
                                                params=Params({'pretrained_file': self.glove_embeddings_file,
                                                               'embedding_dim': embedding_dim,
                                                               'trainable': False,
                                                               'padding_index': 0
                                                              }))

            return pretrain_emb_embedder , pretrain_emb_embedder.get_output_dim(), BasicTextFieldEmbedder({'tokens':pretrain_emb_embedder})
