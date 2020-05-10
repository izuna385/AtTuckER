import torch
import numpy as np
import copy
from tqdm import tqdm
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
import pdb
import json, pickle, joblib
from allennlp.models import Model
import torch.nn as nn

class EvaluatorClass:
    def __init__(self, args, model, vocab, er_vocab, all_entity_num, entity_dim):
        self.args = args
        self.evaluate_on_cpu = self.args.evaluate_on_cpu
        self.is_cuda_available = torch.cuda.is_available()
        self.er_vocab = er_vocab
        self.cuda_device = int(args.cuda_device)
        self.model = model
        self.model.eval()
        self.model.evaluate_flag += 1
        self.sequence_iterator = BasicIterator(batch_size=args.batch_size)
        self.sequence_iterator.index_with(vocab)
        self.all_entity_num = all_entity_num
        self.entity_dim = entity_dim

    def evaluation(self, ds):
        pred_generator = self.sequence_iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        if self.evaluate_on_cpu:
            self.model.cpu()
        pred_generator_tqdm = tqdm(pred_generator, total=self.sequence_iterator.get_num_batches(ds))

        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        with torch.no_grad():
            for batch in pred_generator_tqdm:
                if self.evaluate_on_cpu == False:
                    batch = nn_util.move_to_device(batch, self.cuda_device)
                    preds = self.model.eval_all_entities(batch['head'], batch['relation'])
                    # batch * all ent size
                    for j in range(batch['head'].size(0)):
                        filt = self.er_vocab[(batch['head'][j][0].int().item(), batch['relation'][j][0].int().item())]
                        target_value = preds[j, batch['tail'][j][0].int().item()].item()
                        preds[j, filt] = 0.0
                        preds[j, batch['tail'][j][0].int().item()] = target_value

                    sort_values, sort_idxs = torch.sort(preds, dim=1, descending=True)
                    sort_idxs = sort_idxs.cpu().numpy()
                    for j in range(batch['head'].size(0)):
                        rank = np.where(sort_idxs[j] == batch['tail'][j][0].int().item())[0][0]
                        ranks.append(rank + 1)

                        for hits_level in range(10):
                            if rank <= hits_level:
                                hits[hits_level].append(1.0)
                            else:
                                hits[hits_level].append(0.0)

        print('\n ###### RESULTS ######')
        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        print('###### ###### ######')

        print('\n### EVALUATION FINISHED ###\n')

    def add_Embclass_2_model(self, E_numpy_weight):
        # emb_d, emb_ent_size = E_numpy_weight.shape(0) , E_numpy_weight.shape(1)
        embedding_tensor = torch.Tensor(E_numpy_weight)
        embedding_tensor = embedding_tensor.float()
        if self.is_cuda_available and self.evaluate_on_cpu == False:
            embedding_tensor = embedding_tensor.cuda()
        embed_for_model = nn.Embedding.from_pretrained(embedding_tensor)
        self.model.embed_for_model = embed_for_model # add class variable to model

    def get_E_numpy_from_alldataset(self, dslist):
        entity_symbolidx_2_KGemb_through_linear = {}

        for ds in dslist:
            pred_generator = self.sequence_iterator(ds, num_epochs=1, shuffle=False)
            self.model.eval()
            if self.evaluate_on_cpu:
                self.model.cpu()
            pred_generator_tqdm = tqdm(pred_generator, total=self.sequence_iterator.get_num_batches(ds))

            with torch.no_grad():
                for batch in pred_generator_tqdm:
                    if self.evaluate_on_cpu == False:
                        batch = nn_util.move_to_device(batch, self.cuda_device)
                        hidx, hvec, tidx, tvec = self._extract_head_or_tail_and_its_vectordata(batch)
                        # batch, batch * dim, batch, batch * dim
                        for head_idx, head_ent_vect in zip(hidx, hvec):
                            if head_idx not in entity_symbolidx_2_KGemb_through_linear:
                                entity_symbolidx_2_KGemb_through_linear.update({head_idx:head_ent_vect})
                        for tail_idx, tail_ent_vect in zip(tidx, tvec):
                            if tail_idx not in entity_symbolidx_2_KGemb_through_linear:
                                entity_symbolidx_2_KGemb_through_linear.update({tail_idx: tail_ent_vect})

        E = self.entity_symbolidx_2_KGemb_through_linear__2__E(entity_symbolidx_2_KGemb_through_linear)
        return E

    def entity_symbolidx_2_KGemb_through_linear__2__E(self,entity_symbolidx_2_KGemb_through_linear):
        # pdb.set_trace()
        KBemb = np.zeros((self.all_entity_num, self.entity_dim)).astype('float32')
        for ent_idx, vec in entity_symbolidx_2_KGemb_through_linear.items():
            KBemb[ent_idx] = vec
        print('converted emb', len(entity_symbolidx_2_KGemb_through_linear),'/',self.all_entity_num)
        # pdb.set_trace()
        return KBemb

    ### misc ###

    def idx2int_tensor(self, data):
        return data.int().cpu().detach().numpy()

    def vector2tensor(self, data):
        return data.cpu().detach().numpy()

    def get_W_numpy(self):
        return self.model.get_W()

    def get_R_numpy(self):
        return self.model.get_R()

    def tonp(self,tsr):
        return tsr.detach().cpu().numpy()

    def _extract_head_or_tail_and_its_vectordata(self, batch) -> np.ndarray:
        '''
        :param batch:
        :return: Embedding matrix of all entities.
        '''
        out_dict = self.model(**batch)
        head_idx = self.idx2int_tensor(out_dict['heads']).squeeze()
        tail_idx = self.idx2int_tensor(out_dict['tails']).squeeze()
        head_sents = self.vector2tensor(out_dict['heads_sent_encoded2KGemb'])
        tail_sents = self.vector2tensor(out_dict['tails_sent_encoded2KGemb'])

        return head_idx, head_sents, tail_idx, tail_sents

