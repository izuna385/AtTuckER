import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, pickle
import gc
import pdb
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
# from allennlp.modules.feedforward import Feedforward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.common import Params
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import BertPooler
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
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder,PytorchSeq2SeqWrapper,Seq2SeqEncoder,IntraSentenceAttentionEncoder, QaNetEncoder
from allennlp.modules.seq2seq_encoders import MultiHeadSelfAttention, PassThroughEncoder,GatedCnnEncoder, BidirectionalLanguageModelTransformer, FeedForwardEncoder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder

class DefinitionSentenceEncoder(Seq2VecEncoder):
    def __init__(self, args, input_dim, hidden_dim, word_embedder):
        super(DefinitionSentenceEncoder,self).__init__()
        self.config = args
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.projection_dim = input_dim
        self.feedforward_hidden_dim = input_dim
        self.num_layers = self.args.num_layers_for_stackatt
        self.num_attention_heads = self.args.num_atthead_for_stackatt

        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

        self.mentiontransformer = StackedSelfAttentionEncoder(input_dim=input_dim, hidden_dim=hidden_dim,
                                                       projection_dim=self.projection_dim, feedforward_hidden_dim=self.feedforward_hidden_dim,
                                                       num_layers=self.num_layers, num_attention_heads=self.num_attention_heads)

        self.senttransformer = StackedSelfAttentionEncoder(input_dim=input_dim, hidden_dim=hidden_dim,
                                                       projection_dim=self.projection_dim, feedforward_hidden_dim=self.feedforward_hidden_dim,
                                                       num_layers=self.num_layers, num_attention_heads=self.num_attention_heads)

        self.ff_seq2vecs = nn.Linear(input_dim  ,input_dim)

        self.rnn = PytorchSeq2VecWrapper(
            nn.LSTM(bidirectional=True, num_layers=2, input_size=input_dim, hidden_size=hidden_dim // 2, batch_first=True,
                    dropout=self.args.lstmdropout))

        self.bow = BagOfEmbeddingsEncoder(input_dim, self.args.bow_avg)

    # https://github.com/Monkey911/nlp_hw6/blob/ecd7fe2f4b5b355ca1dde8d90c6bc4cbf1587aab/elmo.py
    # https://github.com/jjccooooll12/Transformer-Allennlp/tree/9ef64bbcf8b774d605da5bd29cfdfb315a1e0148

    def forward(self, sentence, sentence_length):
        if self.args.encoder == 'bow':
            sentence_emb = self.word_embedder(sentence)
            sentence_emb = self.word_embedding_dropout(sentence_emb)
            return self.bow(sentence_emb)

        mask_sentence = get_text_field_mask(sentence)
        sentence_emb = self.word_embedder(sentence)
        sentence_emb = self.word_embedding_dropout(sentence_emb)
        if self.args.encoder == 'lstm':
            sentence_emb = self.rnn(sentence_emb, mask_sentence)

        elif self.args.definition_sentence_max_mean_pool == 'max':
            sentence_emb = torch.max(self.senttransformer(sentence_emb,mask_sentence), dim=1)
        elif self.args.definition_sentence_max_mean_pool == 'mean':

            sentence_emb = sentence_emb.sum(1) / sentence_length

        local_context = self.ff_seq2vecs(sentence_emb)

        return local_context

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim
    #
    @overrides
    def get_output_dim(self) -> int:
        return self.input_dim
    #

####
class RelationAttendedDefinitionSentenceEncoder(Seq2VecEncoder):
    def __init__(self, args, input_dim, hidden_dim, word_embedder):
        super(RelationAttendedDefinitionSentenceEncoder,self).__init__()
        self.config = args
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.projection_dim = input_dim
        self.feedforward_hidden_dim = input_dim
        self.num_layers = self.args.num_layers_for_stackatt
        self.num_attention_heads = self.args.num_atthead_for_stackatt

        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

        # from allennlp.modules.seq2seq_encoders import , , \
        #     , ,
        #     BidirectionalLanguageModelTransformer, FeedForwardEncoder

        if self.args.definition_seq2seq == 'passthrough':
            self.seq2seq = PassThroughEncoder(input_dim=input_dim)
        elif self.args.definition_seq2seq == 'multiheadstackatt':
             self.seq2seq =  StackedSelfAttentionEncoder(input_dim=input_dim, hidden_dim=input_dim,
                                            projection_dim=input_dim, feedforward_hidden_dim=input_dim,
                                            num_layers=2, num_attention_heads=2)
        elif self.args.definition_seq2seq == 'qanet':
            self.seq2seq = QaNetEncoder(input_dim=input_dim, hidden_dim=input_dim,attention_projection_dim=input_dim, feedforward_hidden_dim=input_dim,
                                        num_blocks=2, num_convs_per_block=2,conv_kernel_size=3, num_attention_heads=2)
        elif self.args.definition_seq2seq == 'intrasentenceatt':
            self.seq2seq = IntraSentenceAttentionEncoder(input_dim=input_dim, projection_dim=input_dim, output_dim=input_dim)
        elif self.args.definition_seq2seq == 'gatedcnn':
            self.seq2seq = GatedCnnEncoder(input_dim=512 ,layers=[ [[4, 512]], [[4, 512], [4, 512]], [[4, 512], [4, 512]], [[4, 512], [4, 512]] ] ,dropout=0.05)
        elif self.args.definition_seq2seq == 'bilmtransformer':
            self.seq2seq = BidirectionalLanguageModelTransformer(input_dim=input_dim, hidden_dim=input_dim, num_layers=2)
        # elif self.args.definition_seq2seq == 'feedfoward':
        #     feedforward = FeedForward(input_dim=input_dim, num_layers=1, hidden_dims=input_dim, activations=self.args.activation_for_sentence_ff)
        #     self.seq2seq = FeedForwardEncoder(feedforward)

        # '''
        # *"linear"
        # *`"relu" < https: // pytorch.org / docs / master / nn.html  # torch.nn.ReLU>`_
        # *`"relu6" < https: // pytorch.org / docs / master / nn.html  # torch.nn.ReLU6>`_
        # *`"elu" < https: // pytorch.org / docs / master / nn.html  # torch.nn.ELU>`_
        # *`"prelu" < https: // pytorch.org / docs / master / nn.html  # torch.nn.PReLU>`_
        # *`"leaky_relu" < https: // pytorch.org / docs / master / nn.html  # torch.nn.LeakyReLU>`_
        # *`"threshold" < https: // pytorch.org / docs / master / nn.html  # torch.nn.Threshold>`_
        # *`"hardtanh" < https: // pytorch.org / docs / master / nn.html  # torch.nn.Hardtanh>`_
        # *`"sigmoid" < https: // pytorch.org / docs / master / nn.html  # torch.nn.Sigmoid>`_
        # *`"tanh" < https: // pytorch.org / docs / master / nn.html  # torch.nn.Tanh>`_
        # *`"log_sigmoid" < https: // pytorch.org / docs / master / nn.html  # torch.nn.LogSigmoid>`_
        # *`"softplus" < https: // pytorch.org / docs / master / nn.html  # torch.nn.Softplus>`_
        # *`"softshrink" < https: // pytorch.org / docs / master / nn.html  # torch.nn.Softshrink>`_
        # *`"softsign" < https: // pytorch.org / docs / master / nn.html  # torch.nn.Softsign>`_
        # *`"tanhshrink" < https: // pytorch.org / docs / master / nn.html  # torch.nn.Tanhshrink>`_
        # '''

        elif self.args.definition_seq2seq == 'multiheadselfatt':
            self.seq2seq = MultiHeadSelfAttention(num_heads=2,input_dim=input_dim, output_projection_dim=input_dim, attention_dim=input_dim, values_dim=input_dim)
        else:
            print('Encoder not defined:',self.args.definition_seq2seq)
            exit()

    # https://github.com/Monkey911/nlp_hw6/blob/ecd7fe2f4b5b355ca1dde8d90c6bc4cbf1587aab/elmo.py
    # https://github.com/jjccooooll12/Transformer-Allennlp/tree/9ef64bbcf8b774d605da5bd29cfdfb315a1e0148

    def forward(self, sentence, sentence_length):
        mask_sentence = get_text_field_mask(sentence)
        sentence_emb = self.word_embedder(sentence)
        sentence_emb = self.word_embedding_dropout(sentence_emb)
        sentence_emb = self.seq2seq(sentence_emb,mask_sentence)

        return sentence_emb, mask_sentence

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim
    #
    @overrides
    def get_output_dim(self) -> int:
        return self.input_dim
