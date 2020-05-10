import json, re, pdb
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as multi

# For multiprocessing#
with open('./../../PyTorch-BigGraph/data/relaonly/dictionary.json', 'r') as TBd:
    global_entity_symbols_appear_in_relagraph = json.load(TBd)["entities"]["all"]

with open('./misc_data/originaldata/data/cui2def_tokenized.json', 'r') as cp:
    global_entity_symbols_which_has_definitions = json.load(cp)

global_entity_symbols_which_has_definition = []
for entity_symbols_and_definitions in global_entity_symbols_which_has_definitions:
    for symbol, definition in entity_symbols_and_definitions.items():
        global_entity_symbols_which_has_definition.append(symbol)
global_entity_symbols_which_has_definition = list(set(global_entity_symbols_which_has_definition))
# For multiprocessing#

def add_reverse_writer():
    dataset_path = ['dbpedia50', 'dbpedia500', 'FB15k', 'FB15k-237', 'WN18', 'WN18RR']
    train_valid_test = ['train.txt', 'valid.txt', 'test.txt']

    for flag in train_valid_test:
        for dataset in dataset_path:
            with open('./data/' + dataset + '/' + flag, 'r') as reader, open('./data/' + dataset + '/add_reverse_' + flag, 'w') as writer:
                for line in tqdm(reader):
                    line = line.strip()
                    head = line.split('\t')[0]
                    rel = line.split('\t')[1]
                    tail = line.split('\t')[2]

                    raw_triplet = head + '\t' + rel + '\t' + tail + '\n'
                    add_reverse_triplet = tail + '\t' + rel+'_reverse' + '\t' + head + '\n'

                    writer.write(raw_triplet)
                    writer.write(add_reverse_triplet)

def construct_adjgraph():
    # TODO: Implement
    return 0

def in_rela_cuis_how_many_entity_symbols_have_definition_checker(cui2def_path, TorchBigGraph_dictpath):
    with open(cui2def_path, 'r') as cp:
        entity_symbols_which_has_definitions = json.load(cp)
    with open(TorchBigGraph_dictpath, 'r') as TBd:
        entity_symbols_appear_in_relagraph = json.load(TBd)["entities"]["all"]

    definition_has_entities = []
    for entity_symbols_and_definitions in entity_symbols_which_has_definitions:
        for symbol, definition in entity_symbols_and_definitions.items():
            definition_has_entities.append(symbol)
    definition_has_entities = list(set(definition_has_entities))
    entity_symbols_appear_in_relagraph = list(set(entity_symbols_appear_in_relagraph))

    rela_entity_symbols_which_has_definitions = 0
    for entity_symbol in tqdm(entity_symbols_appear_in_relagraph):
        if entity_symbol in definition_has_entities:
            rela_entity_symbols_which_has_definitions += 1
    print('in rela graph entity symbols which has definitions', rela_entity_symbols_which_has_definitions,'/',len(entity_symbols_appear_in_relagraph))


def multiprocess_rela_entity_symbol_checker_which_has_definition(entity_symbol_list_in_rela): # entity_symbol_list_in_rela
    '''
    :param
    :return:
    '''
    entity_symbol_list_in_rela = list(set(entity_symbol_list_in_rela))
    n_cores = multi.cpu_count()
    print('cpucores', n_cores)
    pools = Pool(n_cores)
    print('multiprocess entity squeezing, start')
    # pdb.set_trace()
    with pools as multipool:
        all_symbol_counts = len(entity_symbol_list_in_rela)
        imap = multipool.imap(entity_symbol_checker_if_entity_symboll_has_definition, entity_symbol_list_in_rela)
        squeezed_count = list(tqdm(imap, total=all_symbol_counts))
    print('\nmultiprocess entity squeezing, finished!')
    entity_count_which_has_Definitions_in_rela_file = 0
    for entity in squeezed_count:
        if entity != '-1':
            entity_count_which_has_Definitions_in_rela_file += 1
    print(' in rela file entities which has definitions,', entity_count_which_has_Definitions_in_rela_file, '/', len(squeezed_count))


def entity_symbol_checker_if_entity_symboll_has_definition(entity_symbol):
    '''
    :param l: [head,rel,tail]
    '''
    if entity_symbol in global_entity_symbols_which_has_definition:
        return entity_symbol
    else:
        return '-1'


if __name__ == '__main__':
    # add_reverse_writer() # This was done for adding reverse relation to all dataset

    cui2def_path = './misc_data/originaldata/data/cui2def_tokenized.json'
    TorchBigGraph_dictpath = './../../PyTorch-BigGraph/data/1000k_relaonly/dictionary.json' # ["entities"]["all"]

    # in_rela_cuis_how_many_entity_symbols_have_definition_checker(cui2def_path,TorchBigGraph_dictpath)
    multiprocess_rela_entity_symbol_checker_which_has_definition(global_entity_symbols_appear_in_relagraph)
