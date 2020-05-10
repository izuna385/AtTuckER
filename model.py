import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, pickle
import gc
import pdb
# from allennlp.modules import FeedForward
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
from torch.nn.init import xavier_normal_
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
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
from torch.nn import functional as F
import faiss
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm

class InBatchDefinitionEncTucker(Model):
    def __init__(self, args,
                 word_embeddings: TextFieldEmbedder,
                 encoder : Seq2VecEncoder,
                 relation_nums : int,
                 d1 : int,
                 d2 : int,
                 vocab,
                 er_vocab):
        super().__init__(vocab)
        self.args = args
        self.encoder = encoder # DefinitionSentenceEncoder
        self.R = torch.nn.Embedding(relation_nums, d2)  # .to('cuda:2')

        if torch.cuda.is_available():
            self.R.cuda()

        if torch.cuda.is_available():
            self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                     dtype=torch.float, device="cuda",
                                                     requires_grad=True))  # .to('cuda:3')
        else:
            self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                     dtype=torch.float, device="cpu", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(self.args.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(self.args.hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(self.args.hidden_dropout2)
        # self.loss = torch.nn.BCELoss()

        self.loss_BCE = torch.nn.BCELoss()
        self.loss_Marginranking = nn.MarginRankingLoss(margin=self.args.margin_for_marginrankloss)
        # https://github.com/TanyaZhao/groupchat_pytorch/blob/4883638769abddc8030d234ec59f01a2d29907e9/class_loss.py

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

        self.def2kgdim = torch.nn.Linear(self.encoder.get_output_dim(), d1)
        self.accuracy = CategoricalAccuracy()

        self.loss_hinge = nn.HingeEmbeddingLoss(margin=self.args.margin_for_hingeemb)

        self.R_proj2worddim_for_relatt = torch.nn.Linear(d2,self.encoder.get_output_dim())
        self.conv = nn.Conv2d(in_channels=relation_nums,
                              out_channels=relation_nums,
                              kernel_size=(self.args.ngram-1, 1),
                              padding=((self.args.ngram-1) // 2, 0)
                              )

        self.att_conv = nn.Conv1d(relation_nums,relation_nums,kernel_size=(self.args.ngram-1, 1),
                              padding=((self.args.ngram-1) // 2, 0))
        self.er_vocab = er_vocab
        self.evaluate_flag = 0

    def init(self):
        xavier_normal_(self.R.weight.data)

    def forward(self, head, head_ent_sent, relation, tail, tail_ent_sent, head_sentence_length, tail_sentence_length, head_cano, tail_cano, tail_golds):
        output = {}
        tmp_batch_size = head.size(0)
        if self.evaluate_flag:
            output['heads'], output['tails'] = head, tail

        if not self.args.only_consider_its_one_gold:
            desired_tail_golds_max_pad = tail_golds.size(1)

            # For gpu conservation, do this permutation on cpu
            index = torch.rand(tmp_batch_size, tmp_batch_size).mul(desired_tail_golds_max_pad).long()# .cpu()
            tail_golds = tail_golds.long()# .cpu()

            expanded_size = (tmp_batch_size,tmp_batch_size,desired_tail_golds_max_pad)
            expanded_index = index.unsqueeze(2).expand(expanded_size)# .cpu()
            expanded_desired_tail_golds = tail_golds.unsqueeze(1).expand(expanded_size)# .cpu()
            if torch.cuda.is_available():
                expanded_index = expanded_index.cuda()
                expanded_desired_tail_golds = expanded_desired_tail_golds.cuda()
            mask = expanded_index.eq(expanded_desired_tail_golds.long()).any(-1)# .cpu()
            index[mask] = -1

            # Diagonal is each triplets's gold.
            ind = np.diag_indices(tmp_batch_size)
            index[ind[0], ind[1]] = - torch.ones(tmp_batch_size).long()
            index[index != -1] = 0
            index[index == -1] = 1
            gold = index.float() # torch.eye(tmp_batch_size)
        else:
            # For
            gold = torch.eye(tmp_batch_size)

        if torch.cuda.is_available():
            gold = gold.cuda()
        # pdb.set_trace()
        if torch.cuda.is_available():
            relation = relation.cuda()
        relation = relation.squeeze(1)

        if self.args.relation_attention2headandtaildef:
            head_sent_sequence, head_sent_sequence_mask = self.encoder(head_ent_sent, head_sentence_length)
            tail_sent_sequence, tail_sent_sequence_mask = self.encoder(tail_ent_sent, tail_sentence_length)

            if self.args.relation_labelatt_enc == 'seq2relatt_1':

                head_sent, head_att = self.seq2relatt_1(head_sent_sequence, head_sent_sequence_mask)
                tail_sent, tail_att = self.seq2relatt_1(tail_sent_sequence, tail_sent_sequence_mask)

            else:
                assert self.args.relation_labelatt_enc == 'seq2relatt_2'
                head_sent, head_att = self.seq2relatt_2(head_sent_sequence, head_sent_sequence_mask)
                tail_sent, tail_att = self.seq2relatt_2(tail_sent_sequence, tail_sent_sequence_mask)

        else:
            head_sent = self.encoder(head_ent_sent, head_sentence_length)
            tail_sent = self.encoder(tail_ent_sent, tail_sentence_length)

        # We can get here, head/tail:encoded sentence raw

        head_sent = self.def2kgdim(head_sent)
        tail_sent = self.def2kgdim(tail_sent)

        if self.evaluate_flag:
            output['heads_sent_encoded2KGemb'] = head_sent
            output['tails_sent_encoded2KGemb'] = tail_sent

        head = self.bn0(head_sent)              # batch * dim, 128 * 200
        head = self.input_dropout(head)     # batch * dim, 128 * 200
        head = head.view(-1, 1, head.size(1)) # batch * 1 * dim, 128 * 1 * 200

        tail = self.bn0(tail_sent)              # batch * dim, 128 * 200
        tail = self.input_dropout(tail)     # batch * dim, 128 * 200
        tail = tail.view(-1, 1, tail.size(1)) # batch * 1 * dim, 128 * 1 * 200

        r = self.R(relation.long())

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, head_sent.size(1), head_sent.size(1)) # batch * dim * dim, 128 * 200 * 200
        W_mat = self.hidden_dropout1(W_mat)                          #  batch * dim * dim, 128 * 200 * 200

        # e1, x, r, W_mat = e1.cuda(), x.cuda(), r.cuda(), W_mat.cuda()

        head = torch.bmm(head, W_mat)
        '''
        dim_ = dim = 200
        x(orig) : batch * 1 * dim_
        W_mat   : batch * dim_ * dim
        x = batch * 1 * dim
        '''

        head = head.view(-1, head_sent.size(1))      # x: batch * dim
        head = self.bn1(head)                 # same
        head = self.hidden_dropout2(head)     # same

        tail = tail.squeeze(1)

        batch_size = tail.size(0)
        ent_d = tail.size(1)

        inbatch_bmm = torch.bmm(tail.squeeze(1).repeat(batch_size,1).view(batch_size,batch_size,ent_d), head.unsqueeze(2)).squeeze(2)

        inbatch_bmm = torch.where(torch.isnan(inbatch_bmm), torch.zeros_like(inbatch_bmm), inbatch_bmm)
        inbatch_bmm = torch.sigmoid(inbatch_bmm)
        eps = 1e-7
        inbatch_bmm = inbatch_bmm.clamp(min=eps, max=1-eps)
        # if there is another gold, switch this to 1
        if torch.cuda.is_available():
            gold = gold.cuda()
        # gold = gold.clamp(min=0, max=1)
        if self.args.loss == 'marginranking':
            golds = gold * 2 - 1 # convert 0,1 --> -1, 1
            # pdb.set_trace()
            heads = head.repeat(batch_size,1).view(batch_size,batch_size,-1)
            tails = tail.repeat(batch_size,1).view(batch_size,batch_size,-1)
            # pdb.set_trace()
            output['loss'] = self.loss_Marginranking(heads.view(-1,200),tails.view(-1,200),golds.view(-1).unsqueeze(-1)).clamp(min=-10000, max=10000)

        elif self.args.loss == 'bce':

            #pdb.set_trace()
            output['loss'] = self.loss_BCE(inbatch_bmm, gold).clamp(min=-10000, max=10000)

        elif self.args.loss == 'hingeemb':
            gold = gold * 2 - 1 # convert 0,1 --> -1, 1
            # pdb.set_trace()
            heads = head.repeat(batch_size,1).view(batch_size,batch_size,-1)
            tails = tail.repeat(batch_size,1).view(batch_size,batch_size,-1)
            score = self.calc_inbatchL1distance(heads,tails)
            output['loss'] = self.loss_hinge(score, gold).clamp(min=-10000, max=10000)

        else:
            print('Loss is not defined')
            exit()

        # pdb.set_trace()
        # self.accuracy(inbatch_bmm, )
        self.accuracy(inbatch_bmm, torch.tensor([i for i in range(batch_size)]))
        return output

    @overrides
    # https://github.com/huggingface/hmtl/blob/5e94a14efb0a0b186563aaf20a3079f99f0a05e0/hmtl/models/layerCoref.py
    # https://github.com/j6mes/fever2-baseline/blob/7db9d6955719e35e6e737339263b8ef20e4cca0e/src/attack/checkmodel.py
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def calc_inbatchL1distance(self,h,t):
        absdiff = torch.abs(h-t) # batch * cand * dim
        return torch.sum(absdiff,dim=2) # batch * cands

    ###################################
    ## seq2relatt_1 implementation https://github.com/cby201432/LEAM.git
    def partial_softmax(self, logits, weights, dim):
        exp_logits = torch.exp(logits)
        # pdb.set_trace()
        exp_logits_weighted = torch.mul(exp_logits, weights.float())
        exp_logits_sum = torch.sum(exp_logits_weighted, dim, True)
        partial_softmax_score = torch.div(exp_logits_weighted, exp_logits_sum)
        return partial_softmax_score

    def relatt_emb_ngram_encoder_maxout(self, x_emb, x_mask):
        """
        output
        """
        # pdb.set_trace()
        x_mask = x_mask.unsqueeze(-1)  # b * s * 1
        x_emb_1 = torch.mul(x_emb, x_mask.float())  # b * s * e

        x_emb_norm = F.normalize(x_emb_1, p=2, dim=2)  # b * s * e
        W_class_norm = F.normalize(self.R_proj2worddim_for_relatt(self.R.weight), p=2, dim=1)  # c * e
        print('einsum seem to have bugs, please shutdown')
        print('See ')
        exit()
        G = torch.einsum("abc,cd->abd", (x_emb_norm, W_class_norm.t()))  # b * s * c
        G = G.permute(0, 2, 1) # b * c * s
        G = G.unsqueeze(-1) # b * c * s * 1 # [64, 442, 23, 1]
        # pdb.set_trace()
        Att_v = F.relu(self.conv(G))  # b * c * s *  1
        Att_v = Att_v.squeeze(-1) # b * c * s
        Att_v = Att_v.permute(0, 2, 1) # b * s * c

        Att_v = torch.max(Att_v, -1, True)[0]  # b * s * 1
        Att_v_max = self.partial_softmax(Att_v, x_mask, 1)  # b * s * 1

        x_att = torch.mul(x_emb, Att_v_max)  # b * s * e
        H_enc = torch.sum(x_att, 1)  # b * e
        return H_enc, Att_v_max

    def seq2relatt_1(self,sentence_emb, sentence_mask):
        H_enc, Att_v_max = self.relatt_emb_ngram_encoder_maxout(sentence_emb, sentence_mask)
        return H_enc, Att_v_max
    ### seq2relatt_1 implementation end
    ###################################


    ### LEAM2 from https://github.com/voghoei/DL-Fundation/blob/06295703c56fcaa300ec85400db0c62aee01c8a7/MH-Term-Project-master/src/LEAM/model.py

    def seq2relatt_2(self, sentence_emb, sentence_mask):
        '''
        :param sentence_emb:
        :param sentence_mask:
        :return:
        NOTE: @2019/11/20 under implementing now
        '''
        W_class_tran = torch.t(self.R_proj2worddim_for_relatt(self.R.weight.data))
        x_mask = torch.unsqueeze(sentence_mask.clone().detach(),-1) #b*s*1
        x_vectors_1 = torch.mul(sentence_emb, x_mask.float())  # b*s*e

        x_vectors_norm = F.normalize(x_vectors_1, p=2, dim=2) #b*s*e
        W_class_norm = F.normalize(W_class_tran, p=2, dim = 0) #e*c
        G = torch.matmul(x_vectors_norm, W_class_norm) #b*s*c
        G = G.permute(0, 2, 1) #b*c*s
        x_full_vectors = sentence_emb

        # pdb.set_trace()
        Att_v = self.att_conv(G.unsqueeze(-1)).clamp(min=0).squeeze(-1)  # b*c*s
        Att_v = Att_v.permute(0, 2, 1)

        # Max pooling
        Att_v, indx = torch.max(Att_v, dim=-1, keepdim=True)

        exp_logits = torch.exp(Att_v)
        exp_logits_masked = torch.mul(exp_logits, x_mask.float())
        exp_logits_sum = torch.sum(exp_logits_masked, dim=1)
        exp_logits_sum = torch.unsqueeze(exp_logits_sum,1)
        partial_softmax_score = torch.div(exp_logits_masked, exp_logits_sum)

        # Get attentive weight
        x_att = torch.mul(x_full_vectors, partial_softmax_score)
        H_enc = torch.sum(x_att, dim=1)
        H_enc = torch.squeeze(H_enc)

        return H_enc, Att_v

    def get_R(self):
        return self.R.weight.data.detach().cpu().numpy()

    def get_W(self):
        return self.W.detach().cpu().numpy()

    def get_er_vocab(self):
        return  self.er_vocab

    def eval_all_entities(self, e1_idx, r_idx):
        '''
        :return: pred of all entity idxs e.g. preds
        embed_for_model will be constructed after train finished
        '''
        e1 = self.embed_for_model(e1_idx.squeeze().long())
        r_idx = r_idx.squeeze().long()
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))
        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))  # batch * dim * dim, 128 * 200 * 200
        W_mat = self.hidden_dropout1(W_mat)  # batch * dim * dim, 128 * 200 * 200

        x = torch.bmm(x, W_mat)  # batch * dim , batch * dim * dim --> batch * 1 * dim

        x = x.view(-1, e1.size(1))  # x: batch * dim
        x = self.bn1(x)  # same
        x = self.hidden_dropout2(x)  # same
        x = torch.mm(x, self.embed_for_model.weight.transpose(1, 0))
        pred = torch.sigmoid(x)  # 128 * 14541

        # pdb.set_trace()
        return pred



    # TODO
    # Infersent, or use another embedding https://github.com/facebookresearch/InferSent/blob/master/models.py