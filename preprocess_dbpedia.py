import json, re, pdb, pickle
from tqdm import tqdm
from memory_profiler import profile
from multiprocessing import Pool
import multiprocessing as multi
import spacy
import scispacy
from all_parameters import Params
import os
from allennlp.common.tee_logger import TeeLogger, replace_cr_with_newline
from datetime import datetime
from pytz import timezone
import sys

class DBPediaPreprocessor:
    def __init__(self, opts):
        '''
        How to deal dpbedia dataset, see
        https://github.com/bxshi/ConMask/blob/master/tools/generate_evaluation_target_files.py
        :param opts:
        '''
        self.opts = opts
        self.all_train_valid_textpath, self.dbpedia_jsonlpath = self.dbpedia_requiredfile_existchecker()
        if self.opts.jsonl2json_alreadydumped == False:
            self.jsonl2jsonparser()
        if self.opts.entitycheckalreadydone == False:
            self.from_conmaskdataset_convert_train_valid_testtxt()
        if self.opts.parsed_json_squeezing_by_entities_on_db50_and_db500_already == False:
            self.from_appearing_entities_on_db50anddb500_and_parsed_json_get_squeezed_parsed_json()
        self.from_squeezed_parsed_json_2_dump_entity2type_desc_cano()


    def dbpedia_requiredfile_existchecker(self):
        dbpedia_rawcorpus_dir = self.opts.dbpedia_rawcorpus_dir
        misc_dir = self.opts.misc_dir
        dataset = ['dbpedia50', 'dbpedia500']
        train_dev_test_list = ['train.txt', 'valid.txt', 'test.txt']

        required_jsol_path = dbpedia_rawcorpus_dir + 'dbpedia_ents.text.jsonl'
        required_filepaths = list()
        for one_data in dataset:
            for _ in train_dev_test_list:
                required_filepaths.append(misc_dir + one_data + '/' + _)

        missing_fileflag = 0
        for filepath in required_filepaths:
            if not os.path.isfile(filepath):
                missing_fileflag += 1
                print(filepath,'is missing')
        if not os.path.isfile(required_jsol_path):
            missing_fileflag += 1
            print(required_jsol_path, 'is missing')

        if missing_fileflag:
            print('Needed file missing, shutdown')
            exit()
        else:
            return required_filepaths, required_jsol_path

    def dquotation_remover(self,string):
        string = string.replace('\"','')
        return string

    def from_squeezed_parsed_json_2_dump_entity2type_desc_cano(self):
        squeezed_parsed_dict_path = './data/dict_of_entities_of_db50_and_db500.json'

        db50_entity_symbol2canonical_path = './data/dbpedia50/entity_symbol2cano.txt'
        db50_entity_symbol2type_path = './data/dbpedia50/entity_symbol2type.txt'
        db50_entity_symbol2multisentdesc_path = './data/dbpedia50/entity_symbol2multisentdesc.txt'

        db500_entity_symbol2canonical_path = './data/dbpedia500/entity_symbol2cano.txt'
        db500_entity_symbol2type_path = './data/dbpedia500/entity_symbol2type.txt'
        db500_entity_symbol2multisentdesc_path = './data/dbpedia500/entity_symbol2multisentdesc.txt'

        print('squeezed json loading...')
        with open(squeezed_parsed_dict_path, 'r') as sqj:
            entities_on_db50and500 = json.load(sqj)
        print('squeezed json loaded')

        no_cano_ent_str = 0
        no_type_ent_str = 0
        no_desc_ent_count = 0
        with open(db50_entity_symbol2canonical_path, 'w') as db50cano, open(db50_entity_symbol2type_path, 'w') as db50type, open(db50_entity_symbol2multisentdesc_path, 'w') as db50desc:
            with open(db500_entity_symbol2canonical_path, 'w') as db500cano, open(db500_entity_symbol2type_path,'w') as db500type, open(db500_entity_symbol2multisentdesc_path, 'w') as db500desc:
                print('writing entity_symbol2type,cano, multidesc Start')
                for entity_symbol, related_data in tqdm(entities_on_db50and500.items()):
                    entity_canonical = ' '.join(entity_symbol.split('_'))
                    entity_multisentdesc = related_data["context"].split('\\n')[0]
                    entity_type = related_data['label']
                    entity_canonical = self.dquotation_remover(entity_canonical)
                    entity_multisentdesc = self.dquotation_remover(entity_multisentdesc)
                    entity_type = self.dquotation_remover(entity_type)

                    cano_str = entity_symbol + '\t' + entity_canonical + '\n'
                    desc_str = entity_symbol + '\t' + entity_multisentdesc + '\n'
                    type_str = entity_symbol + '\t' + entity_type + '\n'

                    if not len(entity_type.strip()) == 0:
                        db50cano.write(type_str)
                        db500cano.write(type_str)
                    else:
                        no_cano_ent_str += 1
                    if not len(entity_type.strip()) == 0:
                        db50type.write(type_str)
                        db500type.write(type_str)
                    else:
                        no_type_ent_str += 1
                    if not len(entity_multisentdesc.strip()) == 0:
                        db50desc.write(desc_str)
                        db500desc.write(desc_str)
                    else:
                        no_desc_ent_count += 1
                        db50desc.write(type_str)
                        db500desc.write(type_str)
        print('no_cano_ent_str', no_cano_ent_str ,'/' , len(entities_on_db50and500))
        print('no_type_ent_str', no_type_ent_str ,'/' , len(entities_on_db50and500))
        print('no_desc_ent_count', no_desc_ent_count ,'/' , len(entities_on_db50and500))
        '''
        no_cano_ent_str 5 / 490539
        no_type_ent_str 5 / 490539
        no_desc_ent_count 5624 / 490539
        '''
        print('creating entity_symbol2type.txt, entity_symbol2cano.txt and entity_symbol2multisentdesc.txt of dbpedia50/500 dataset FINISHED!')
        print('run python3 preprocess_ent2type_desc_cano_reladj.py -entity_symbol2cano_type_desc_alreadydumped False -reverse_rel_data_alreadydumped False -KBdataset dbpedia50')
        print('run python3 preprocess_ent2type_desc_cano_reladj.py -entity_symbol2cano_type_desc_alreadydumped False -reverse_rel_data_alreadydumped False -KBdataset dbpedia500')

    def from_appearing_entities_on_db50anddb500_and_parsed_json_get_squeezed_parsed_json(self):
        appearing_entitie_path = './data/dbpedia_appearing_entities_amond_dbpedia50_and_dbperia500.txt'
        parsed_json_path = './misc_data/dbpedia2016/parsed.json'
        squeezed_json_path = './data/dict_of_entities_of_db50_and_db500.json'

        squeezed_json = {}
        print('parsed json loaing...')
        with open(parsed_json_path, 'r') as pj, open(appearing_entitie_path, 'r') as ae:
            parsed_json_all = json.load(pj)
            print('parsed json loaded!')
            all_entities_on_db50_anddb500 = list()
            for line in ae:
                all_entities_on_db50_anddb500.append(line.strip())

            if not self.opts.multiprocess:
                for entity, entitydata in tqdm(parsed_json_all.items()):
                    if entity in all_entities_on_db50_anddb500:
                        squeezed_json.update({entity: entitydata})
                with open(squeezed_json_path, 'w') as sjp:
                    json.dump(squeezed_json, sjp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

            else:
                to_bo_multiprocessed_list = list()
                for entity, entitydata in tqdm(parsed_json_all.items()):
                    to_bo_multiprocessed_list.append([entity, entitydata])
                squeezed_json_dict = multiprocess_squeezedentity_finder(to_bo_multiprocessed_list)

                with open(squeezed_json_path, 'w') as sjp:
                    json.dump(squeezed_json_dict, sjp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    def jsonl2jsonparser(self):
        '''
        :return: dump jsonl --> json to same dir

        { surf: {label: ***, context: ***} }
        '''
        dbpedia_rawcorpus_dir = self.opts.dbpedia_rawcorpus_dir
        required_jsol_path = dbpedia_rawcorpus_dir + 'dbpedia_ents.text.jsonl'

        print('jsonl file loading')
        with open(required_jsol_path, 'r') as json_file:
            json_list = list(json_file)
        print('jsonl file loaded!')

        enti2labelandcontext = {}

        for json_str in tqdm(json_list):
            result = json.loads(json_str)

            context = result['context']
            surface = result['surf']
            uri = result['uri']
            label = result['label']

            enti2labelandcontext.update({surface:{'context':context,
                                                  'label':label,
                                                  'uri': uri}})
        with open(dbpedia_rawcorpus_dir + 'parsed.json', 'w') as pj:
            json.dump(enti2labelandcontext, pj, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    def from_conmaskdataset_convert_train_valid_testtxt(self):
        if not os.path.isdir('./data/dbpedia50/'):
            os.mkdir('./data/dbpedia50/')
        if not os.path.isdir('./data/dbpedia500/'):
            os.mkdir('./data/dbpedia500/')

        appearing_entities = list()

        train_valid_test_flag = ['train','valid', 'test']
        db50_or_500 = ['dbpedia50', 'dbpedia500']
        for one_flag in train_valid_test_flag:
            for data in db50_or_500:
                with open('./misc_data/' + data + '/' + one_flag + '.txt', 'r') as read, open('./data/' + data + '/' + one_flag + '.txt', 'w') as dump:
                    for line in tqdm(read):
                        head, tail, rel = line.strip().split('\t')
                        dump_str_to_KBdata = head + '\t' + rel + '\t' + tail + '\n'
                        dump.write(dump_str_to_KBdata)
                        appearing_entities.append(head)
                        appearing_entities.append(tail)
        with open('./data/dbpedia_appearing_entities_amond_dbpedia50_and_dbperia500.txt', 'w') as dap:
            for entity in tqdm(list(set(appearing_entities))):
                dap.write(entity + '\n')
        print('entity check DONE. See ./data/dbpedia_appearing_entities_amond_dbpedia50_and_dbperia500.txt')

    def needed_entityinfo_collector_fron_parsed_jsonl(self):
        '''
        since dbpedia contains lots entities, we have to focus on only needed entitiy to preprocess them.
        :return: collected entities list
        '''
        return 0

##############################
##### Multiprocess won't work inside class, so define this outside class
appearing_entitie_path_for_multi = './data/dbpedia_appearing_entities_amond_dbpedia50_and_dbperia500.txt'
if not os.path.isfile(appearing_entitie_path_for_multi):
    print('first make ',appearing_entitie_path_for_multi, 'by run\n python3 preprocess_dbpedia.py -multiprocess False -jsonl2json_alreadydumped False -entitycheckalreadydone False')
    exit()
with open(appearing_entitie_path_for_multi, 'r') as aem:
    all_entities_on_db50_anddb500_for_multi = list()
    for line in aem:
        all_entities_on_db50_anddb500_for_multi.append(line.strip())

def multiprocess_squeezedentity_finder(entity_surface_and_entity_data_list):
    '''
    :param
    :return:
    '''

    n_cores = multi.cpu_count()
    print('cpucores', n_cores)
    pools = Pool(n_cores)
    print('multiprocess entity squeezing, start')
    with pools as multipool:
        all_symbol_counts = len(entity_surface_and_entity_data_list)
        imap = multipool.imap(db50_500squeezer_from_parsed_json_list, entity_surface_and_entity_data_list)
        squeezed_parsed_data = list(tqdm(imap, total=all_symbol_counts))
    print('\nmultiprocess entity squeezing, finished!')
    symbols_and_relateddata_dict = {}
    for entity_symbol_and_related_data in tqdm(squeezed_parsed_data):
        if entity_symbol_and_related_data != [-999, -999]:
            symbols_and_relateddata_dict.update({entity_symbol_and_related_data[0]:entity_symbol_and_related_data[1]})
    return symbols_and_relateddata_dict

def db50_500squeezer_from_parsed_json_list(l):
    '''
    :param l: [entity surface, its related data]
    :return: if find in all_entities_on_db50_anddb500_for_multi, return [entity surface, its related data]  else return [-999,-999]
    '''
    entity = l[0]
    if entity in all_entities_on_db50_anddb500_for_multi:
        return l
    else:
        return [-999, -999]

def multiprocess_description_parser(to_be_parserd_symbol_and_definiton_list, spacy_model):
    '''
    :param to_be_parserd_symbol_and_definiton_list: [[symbol, raw_not_split_descstr], [symbol, raw_not_split_descstr], ...]
    :return:
    '''
    n_cores = multi.cpu_count()
    print('cpucores', n_cores)
    pools = Pool(n_cores)
    print('multiprocess entity descriptions, start')
    with pools as multipool:
        all_symbol_counts = len(to_be_parserd_symbol_and_definiton_list)
        imap = multipool.imap(single_symbol_and_raw_entity_desc_parser, to_be_parserd_symbol_and_definiton_list)
        parsed_symbol_and_desc_tokenized_sentences = list(tqdm(imap, total=all_symbol_counts))
    print('\nmultiprocess entity descriptions, finished!')
    symbols_and_tokenized_sentences_dict = {}
    for entity_symbol_and_tokenized_sentences in parsed_symbol_and_desc_tokenized_sentences:
        symbols_and_tokenized_sentences_dict.update({entity_symbol_and_tokenized_sentences[0]:entity_symbol_and_tokenized_sentences[1]})
    return symbols_and_tokenized_sentences_dict

def multiprocess_canonical_parser(to_be_parsed_symbol_and_canonical_list, spacy_model):
    '''
    :param to_be_parserd_symbol_and_definiton_list: [[symbol, raw_canonicalstr], [symbol, raw_canonicalstr], ...]
    :return:
    '''
    n_cores = multi.cpu_count()
    print('cpucores', n_cores)
    pools = Pool(n_cores)
    print('multiprocess entity canonical name, start')
    with pools as multipool:
        all_symbol_counts = len(to_be_parsed_symbol_and_canonical_list)
        imap = multipool.imap(single_symbol_and_canonical_parser,to_be_parsed_symbol_and_canonical_list)
        parsed_symbol_and_canonicals = list(tqdm(imap, total=all_symbol_counts))
    print('\n multiporcess entity canonical name finished!')
    symbol_and_canonical_token_dict = {}
    for entity_symbol_and_canonicaltokens in parsed_symbol_and_canonicals:
        symbol_and_canonical_token_dict.update({entity_symbol_and_canonicaltokens[0]:entity_symbol_and_canonicaltokens[1]})

    return symbol_and_canonical_token_dict

def single_symbol_and_raw_entity_desc_parser(symbol_and_desc_list):
    '''
    :param symbol_and_canonical_list: [symbol, desc]
    :return: [symbol, [[sentence1,tokens,are,here],[sentence2, tokens, are, here]... ]
    '''
    symbol = symbol_and_desc_list[0]
    desc = symbol_and_desc_list[1]

    parsed_sentences_of_one_entity = global_spacy_model(desc)

    one_entity_splited_sentences = list()
    for span in parsed_sentences_of_one_entity.sents:
        one_entity_splited_sentences.append(span)

    one_entity_tokenized_sentences = list()
    for sentence in one_entity_splited_sentences:
        tokenized_one_setntence = list()
        parsed_one_sentence = global_spacy_model(sentence.text)
        for token in parsed_one_sentence:
            tokenized_one_setntence.append(token.text)
        one_entity_tokenized_sentences.append(tokenized_one_setntence)

    return [symbol, one_entity_tokenized_sentences]

def single_symbol_and_canonical_parser(symbol_and_canonical_list):
    '''
    :param symbol_and_canonical_list: [symbol, canonical]
    :return: [symbol, [canonical, token, entity, name]]
    '''
    symbol = symbol_and_canonical_list[0]
    canonical = symbol_and_canonical_list[1]
    parsed_canos = global_spacy_model(canonical)
    tokenized_canos = list()
    for token in parsed_canos:
        tokenized_canos.append(token.text)
    return [symbol, tokenized_canos]

# here is for global spacy model
def can_be_sentence_start_forglobalspacy(token, doc):
    if token.i == 0:
        # print('TOKEN I',token, token.i)
        return True
    # We're not checking for is_title here to ignore arbitrary titlecased
    # tokens within sentences
    # elif token.is_title:
    #    return True

    elif str(token.nbor(-1).text + token.nbor(0).text ) in doc.text:
        try:
            if str(token.nbor(-1).text + token.nbor(0).text + token.nbor().text) in doc.text:
                # print(str(token.nbor(-1).text + token.nbor(0).text + token.nbor().text))
                return False
            else:
                pass
        except:
            pass

    elif token.nbor(-1).is_punct:
        return True
    elif token.nbor(-1).is_space:
        return True
    else:
        return False

# here is for global spacy model
def prevent_sentence_boundaries_forglobalspacy(doc):
    for i, token in enumerate(doc):
        if not can_be_sentence_start_forglobalspacy(token, doc):
            token.is_sent_start = False
    return doc
##### multiprocess parser end.

if __name__ == '__main__':
    allParams = Params()
    params_for_dbpedia_preprocess = allParams.get_preprocess_dbpedia()
    global_spacy_model = spacy.load('en_core_web_lg')
    global_spacy_model.add_pipe(prevent_sentence_boundaries_forglobalspacy, before="parser")
    DBPediaPreprocessor = DBPediaPreprocessor(opts=params_for_dbpedia_preprocess)