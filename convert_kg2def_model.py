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

class SimpleKGProjector(Model):
    def __init__(self, args,
                 kgemb_dim,
                 encoder : Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.encoder = encoder
        self.first_and_last_emb = encoder.get_output_dim()
        self.loss = nn.MSELoss()
        self.projector = nn.Linear(kgemb_dim,encoder.get_output_dim())
        self.projector2 = nn.Linear(kgemb_dim, encoder.get_output_dim())
        self.projector3 = nn.Linear(kgemb_dim, encoder.get_output_dim())
        self.nonlinear = nn.Tanhshrink()

    def forward(self, source_kgemb, target_definitions, sentence_length):
        x = self.encoder(target_definitions, sentence_length)
        # proj_x = self.projector3(self.nonlinear(self.projector2(self.nonlinear(self.projector(source_kgemb)))))
        proj_x = self.projector(source_kgemb)
        l2_diff = self.L2_calc_and_filter_before_loss_calc_defend_loss_explosion(x, proj_x)
        print('l2 max, min, mean:', torch.max(l2_diff).item(), torch.min(l2_diff).item(),torch.mean(l2_diff).item())
        x[torch.where(l2_diff > torch.min(l2_diff) * 5)] = torch.zeros(300).cuda()
        proj_x[torch.where(l2_diff > torch.min(l2_diff) * 5)] = torch.zeros(300).cuda()
        print('abnormal tensor num in batch', (torch.where(l2_diff > torch.min(l2_diff) * 5)[0].size(0)),'/',x.size(0))
        output = {}
        output['loss'] = self.loss(proj_x, x) # sometimese, loss explosion happen.
        return output

    def L2_calc_and_filter_before_loss_calc_defend_loss_explosion(self, x1, x2):
        l2_diff = x1 - x2
        return torch.norm(l2_diff, dim=1)

    def projected_kgemb_returner(self,source_kgemb_torchtensor):
        proj_x = self.projector(source_kgemb_torchtensor)
        return proj_x

    def linear_weight_dumper(self):
        return self.projector.weight.detach().cpu().numpy()

class ProjectedKG_AE(Model):
    def __init__(self, args,
                 kgemb,
                 proj_txtemb, vocab):
        super().__init__(vocab)
        self.args = args
        self.loss = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(1024, 900),
            nn.ReLU(True),
            nn.Linear(900, 600),
            nn.ReLU(True),
            nn.Linear(600, 300))

        self.decoder = nn.Sequential(
            nn.Linear(300, 600),
            nn.ReLU(True),
            nn.Linear(600, 900),
            nn.ReLU(True),
            nn.Linear(900, 1024),
            nn.Tanhshrink()
        )

    def forward(self, source_kgemb):
        x_300 = self.encoder(source_kgemb)
        x_origdim = self.decoder(x_300)

        output = {}
        output['loss'] = self.loss(source_kgemb, x_origdim)
        return output

    def compressed_emb_returner(self, source_kgemb):
        return self.encoder(source_kgemb)

class KGemb2def_projector:
    def __init__(self, args, model, vocab, entity_dim):
        self.args = args
        self.model = model

        self.source_entity_symbol2torchkgemb_path = './misc_data/data/include_REL_and_RELA_entity_embeddings.tsv'



class DefinitionVAE(Model):
    def __init__(self, args,
                 kgemb_dim,
                 encoder : Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.encoder = encoder
        self.first_and_last_emb = encoder.get_output_dim()

        self.fc1 = nn.Linear(self.first_and_last_emb, kgemb_dim)
        self.fc21 = nn.Linear(kgemb_dim, 300)
        self.fc22 = nn.Linear(kgemb_dim, 300)
        self.fc3 = nn.Linear(300, kgemb_dim)
        self.fc4 = nn.Linear(kgemb_dim, self.first_and_last_emb)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, source_kgemb, target_definitions,sentence_length):
        x = self.encoder(target_definitions, sentence_length)
        output = {}
        mu, logvar = self.encode(x.view(-1, self.encoder.get_output_dim()))
        z = self.reparameterize(mu, logvar)
        output['loss'] = self.loss_function(self.decode(z), x, mu, logvar)
        return output #self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy( x.view(-1,  self.encoder.get_output_dim()), recon_x,  reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
