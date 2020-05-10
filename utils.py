import os, sys
import json, pickle, joblib, pdb
from all_parameters import Params
import time
from datetime import datetime
#import request
import numpy as np
import gc
import copy
from tqdm import tqdm
from datetime import datetime
from allennlp.common.tee_logger import TeeLogger # Logging with tee
from pytz import timezone


def string_factorlist2idxdixt_returner(list_which_may_be_include_dups):
    '''
    :param list_which_may_be_include_dups:
    :return: unique_key_list, idx2factor, factor2idx
    '''

    duplicate_deleted_list = list(set(list_which_may_be_include_dups))
    idx2factor = {}
    factor2idx = {}

    for idx, factor in enumerate(duplicate_deleted_list):
        idx2factor.update({idx:factor})
        factor2idx.update({factor:idx})

    return duplicate_deleted_list, factor2idx, idx2factor

def from_KBdatasetdir_return_unique_entities_and_relations_dict(KBdatasetdir_which_has_train_valid_test,
                                                                cached_idxnized_dict_pklpath,
                                                                pkl_file_exists=False,
                                                                reverse_opt=True):
    if pkl_file_exists:
        with open(cached_idxnized_dict_pklpath, 'rb') as cipr:
            cached_idxnized_entity_symbol_and_relations = pickle.load(cipr)
        uniq_entities_in_trn_valid_test = cached_idxnized_entity_symbol_and_relations[
            'uniq_entities_in_trn_valid_test']
        entity_symbol2idx = cached_idxnized_entity_symbol_and_relations['entity_symbol2idx']
        idx2entity_symbol = cached_idxnized_entity_symbol_and_relations['idx2entity_symbol']
        uniq_relations_in_trn_valid_test = cached_idxnized_entity_symbol_and_relations[
            'uniq_relations_in_trn_valid_test']
        relation2idx = cached_idxnized_entity_symbol_and_relations['relation2idx']
        idx2relation = cached_idxnized_entity_symbol_and_relations['idx2relation']
        return uniq_entities_in_trn_valid_test, entity_symbol2idx, idx2entity_symbol, uniq_relations_in_trn_valid_test, relation2idx, idx2relation


    if reverse_opt:
        train_path = KBdatasetdir_which_has_train_valid_test + 'add_reverse_train.txt'
        valid_path = KBdatasetdir_which_has_train_valid_test + 'add_reverse_valid.txt'
        test_path = KBdatasetdir_which_has_train_valid_test + 'add_reverse_test.txt'
    else:
        train_path = KBdatasetdir_which_has_train_valid_test + 'train.txt'
        valid_path = KBdatasetdir_which_has_train_valid_test + 'valid.txt'
        test_path = KBdatasetdir_which_has_train_valid_test + 'test.txt'

    appering_entities_in_train_dev_test_triplets = list()
    appering_relations_in_train_dev_test_triplets = list()

    for dataset in [train_path, valid_path, test_path]:
        with open(dataset,'r') as ds:
            for line in tqdm(ds):
                line = line.strip()
                head = line.split('\t')[0]
                rel = line.split('\t')[1]
                tail = line.split('\t')[2]

                appering_entities_in_train_dev_test_triplets.append(head)
                appering_entities_in_train_dev_test_triplets.append(tail)

                appering_relations_in_train_dev_test_triplets.append(rel)

    appering_entities_in_train_dev_test_triplets = list(set(appering_entities_in_train_dev_test_triplets))
    appering_relations_in_train_dev_test_triplets = list(set(appering_relations_in_train_dev_test_triplets))

    uniq_entities_in_trn_valid_test, entity_symbol2idx, idx2entity_symbol =  string_factorlist2idxdixt_returner(appering_entities_in_train_dev_test_triplets)
    uniq_relations_in_trn_valid_test, relation2idx, idx2relation = string_factorlist2idxdixt_returner(appering_relations_in_train_dev_test_triplets)

    with open(cached_idxnized_dict_pklpath, 'wb') as cip:
        pickle.dump({
            'uniq_entities_in_trn_valid_test': uniq_entities_in_trn_valid_test,
            'entity_symbol2idx': entity_symbol2idx,
            'idx2entity_symbol': idx2entity_symbol,
            'uniq_relations_in_trn_valid_test': uniq_relations_in_trn_valid_test,
            'relation2idx': relation2idx,
            'idx2relation': idx2relation
        }, cip)

    return uniq_entities_in_trn_valid_test, entity_symbol2idx, idx2entity_symbol, uniq_relations_in_trn_valid_test, relation2idx, idx2relation

def from_KBdatasetdir_return_idxnized_triplets_data(KBdatasetdir_which_has_train_valid_test, cached_idxnized_dict_pklpath, cached_indexnized_triplet_trn_valid_test_pklpath,pkl_file_exists=False,reverse_opt=True):
    if pkl_file_exists:
        with open(cached_indexnized_triplet_trn_valid_test_pklpath, 'rb') as cittv:
            all_related_data = pickle.load(cittv)
            train_triplets = all_related_data['train_triplets']
            valid_triplets = all_related_data['valid_triplets']
            test_triplets = all_related_data['test_triplets']
            all_triplets = all_related_data['all_triplets']
            train_triplets_idx = all_related_data['train_triplets_idx']
            valid_triplets_idx = all_related_data['valid_triplets_idx']
            test_triplets_idx = all_related_data['test_triplets_idx']
        return train_triplets, valid_triplets, test_triplets, all_triplets, train_triplets_idx, valid_triplets_idx , test_triplets_idx

    else:
        if reverse_opt:
            train_path = KBdatasetdir_which_has_train_valid_test + 'add_reverse_train.txt'
            valid_path = KBdatasetdir_which_has_train_valid_test + 'add_reverse_valid.txt'
            test_path = KBdatasetdir_which_has_train_valid_test + 'add_reverse_test.txt'
        else:
            train_path = KBdatasetdir_which_has_train_valid_test + 'train.txt'
            valid_path = KBdatasetdir_which_has_train_valid_test + 'valid.txt'
            test_path = KBdatasetdir_which_has_train_valid_test + 'test.txt'

        with open(cached_idxnized_dict_pklpath, 'rb') as cipr:
            cached_idxnized_entity_symbol_and_relations = pickle.load(cipr)
        uniq_entities_in_trn_valid_test = cached_idxnized_entity_symbol_and_relations['uniq_entities_in_trn_valid_test']
        entity_symbol2idx = cached_idxnized_entity_symbol_and_relations['entity_symbol2idx']
        idx2entity_symbol = cached_idxnized_entity_symbol_and_relations['idx2entity_symbol']
        uniq_relations_in_trn_valid_test = cached_idxnized_entity_symbol_and_relations['uniq_relations_in_trn_valid_test']
        relation2idx = cached_idxnized_entity_symbol_and_relations['relation2idx']
        idx2relation = cached_idxnized_entity_symbol_and_relations['idx2relation']

        train_triplets = {}
        valid_triplets = {}
        test_triplets = {}
        all_triplets = {}

        train_triplets_idx = []
        valid_triplets_idx = []
        test_triplets_idx = []

        with open(train_path, 'r') as tp:
            for line in tqdm(tp):
                headidx, relidx, tailidx = line2indexnized_triplet(line, entity_symbol2idx, relation2idx)

                tmp_idx_of_one_triplets = len(all_triplets)
                train_triplets.update({tmp_idx_of_one_triplets:[headidx, relidx, tailidx]})
                all_triplets.update({tmp_idx_of_one_triplets:[headidx, relidx, tailidx]})
                train_triplets_idx.append(tmp_idx_of_one_triplets)

        with open(valid_path, 'r') as tp:
            for line in tqdm(tp):
                headidx, relidx, tailidx = line2indexnized_triplet(line, entity_symbol2idx, relation2idx)

                tmp_idx_of_one_triplets = len(all_triplets)
                valid_triplets.update({tmp_idx_of_one_triplets: [headidx, relidx, tailidx]})
                all_triplets.update({tmp_idx_of_one_triplets:[headidx, relidx, tailidx]})
                valid_triplets_idx.append(tmp_idx_of_one_triplets)

        with open(test_path, 'r') as tp:
            for line in tqdm(tp):
                headidx, relidx, tailidx = line2indexnized_triplet(line, entity_symbol2idx, relation2idx)

                tmp_idx_of_one_triplets = len(all_triplets)
                test_triplets.update({tmp_idx_of_one_triplets: [headidx, relidx, tailidx]})
                all_triplets.update({tmp_idx_of_one_triplets:[headidx, relidx, tailidx]})
                test_triplets_idx.append(tmp_idx_of_one_triplets)

        with open(cached_indexnized_triplet_trn_valid_test_pklpath, 'wb') as citt:
            pickle.dump({
                'train_triplets':train_triplets,
                'valid_triplets':valid_triplets,
                'test_triplets':test_triplets,
                'all_triplets':all_triplets,
                'train_triplets_idx':train_triplets_idx,
                'valid_triplets_idx':valid_triplets_idx,
                'test_triplets_idx':test_triplets_idx,
            }, citt)

        return train_triplets, valid_triplets, test_triplets, all_triplets, train_triplets_idx, valid_triplets_idx , test_triplets_idx


def line2indexnized_triplet(line, entity_symbol2idx, relation2idx):
    line = line.strip()
    head, rel, tail = line.split('\t')
    return entity_symbol2idx[head], relation2idx[rel], entity_symbol2idx[tail]

def entity_symbol2canonical_and_entity_symbol2definitions___2___entity_symbol2cano_and_entity_symbol2def(cached_entity_symbol_idx2canonical_strings_and_entity_definitions_pklpath,
                                                                                                         entity_symbol2idx,
                                                                                                         specified_datasetdir):
    # if os.path.exists(cached_entity_symbol_idx2canonical_strings_and_entity_definitions_pklpath):
    #     with open(cached_entity_symbol_idx2canonical_strings_and_entity_definitions_pklpath, 'rb') as cesl:
    #         related_data = pickle.load(cesl)
    #         return related_data['entsymbolidx2cano'], related_data['entsymbolidx2def']

    entity_symbol2cano_path = specified_datasetdir + 'entity_symbol2cano_unkpad.json'
    entity_symbol2defsentences_path = specified_datasetdir + 'entity_symbol2desc_unkpad.json'

    entsymbolidx2cano = {}
    unk_cano = 0
    entsymbolidx2def = {}
    unk_def_ent = 0
    with open(entity_symbol2cano_path, 'r') as escp:
        symbol2cano = json.load(escp)
    with open(entity_symbol2defsentences_path, 'r') as esdp:
        symbol2desc = json.load(esdp)

    for entity_symbol, idx in entity_symbol2idx.items():
        if entity_symbol in symbol2cano:
            entsymbolidx2cano.update({idx:symbol2cano[entity_symbol] })
        else:
            entsymbolidx2cano.update({idx: ['@@UNKNOWN@@']})
            unk_cano += 1

        if entity_symbol in symbol2desc:
            entsymbolidx2def.update({idx:symbol2desc[entity_symbol] })
        else:
            entsymbolidx2def.update({idx:[['@@UNKNOWN@@']]})
            unk_def_ent += 1
    print('\n##########')
    print('entity symbols which has no canonical', unk_cano ,'/', len(entity_symbol2idx))
    print('entity symbols which has no definition', unk_def_ent, '/', len(entity_symbol2idx))
    print('##########\n')

    with open(cached_entity_symbol_idx2canonical_strings_and_entity_definitions_pklpath, 'wb') as ces:
        pickle.dump({'entsymbolidx2cano':entsymbolidx2cano,
                     'entsymbolidx2def':entsymbolidx2def},ces)
    return entsymbolidx2cano, entsymbolidx2def

def datasetteelogger(KBspecified_experiment_dir):
    experimet_logdir = KBspecified_experiment_dir + 'experiment_logdir/'
    if not os.path.isdir(experimet_logdir):
        os.mkdir(experimet_logdir)
    timestamp = datetime.now(timezone('Asia/Tokyo'))
    str_timestamp = '{0:%Y%m%d_%H%M%S}'.format(timestamp)[2:]

    dir_for_each_experiment = experimet_logdir + str_timestamp
    dir_for_each_experiment += '/'
    loggerpath = dir_for_each_experiment + 'teelog.log'
    os.mkdir(dir_for_each_experiment)
    dir_for_candidate_dumping = dir_for_each_experiment + 'dumped_candidates'
    os.mkdir(dir_for_candidate_dumping)
    print('\n====== ===== =====\nNOTE: TIMESTAMP for this experiment:', dir_for_each_experiment)
    print('====== ===== =====')
    sys.stdout = TeeLogger(loggerpath, sys.stdout, False)  # default: False
    sys.stderr = TeeLogger(loggerpath, sys.stderr, False)  # default: False
    return dir_for_each_experiment

