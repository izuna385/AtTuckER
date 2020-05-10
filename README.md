# AtTuckER

Knowledge Graph Emedding with relation-definition and relation-entity attention.

## Preprocess:
* Check in `./data/`, whether specific dataset exist.

### Preprocessing FB**
```
data--FB15k
    --FB15k237---train.txt
              ---valid.txt
              ---text.txt
              ---entity2multisentdesc.txt
              ---entity2type.txt
              ---entity_symbol2type.txt
```

### NOTE: when preprocessing, you first need

    ```
    train.txt : head_symbol \t relation \t tail+symbol in each one col.
    valid.txt : head_symbol \t relation \t tail+symbol in each one col.
    test.txt : head_symbol \t relation \t tail+symbol  in each one col.
    entity_symbol2cano.txt : entity_symbol \t entity canonical name              in each one col.
    entity_symbol2type.txt : entity_symbol \t type1 \t type2 \t type3 ...        in each one col.
    entity_symbol2multisentdesc.txt : entity_symbol \t multisentence description in each one col.
    ```
Preprocess run command example:
`python3 preprocess_ent2type_desc_cano_reladj.py -entity_symbol2cano_type_desc_alreadydumped False -reverse_rel_data_alreadydumped False -KBdataset FB15k-237 -spacy_model_str en_core_web_lg -multiprocess True`

### Preprocessing WN**

* First, download WordNet-3.0.tar.gz to `./misc_data/` and do `tar -xzvf *`
* run `python3 preprocess_wordnet.py -KBdataset WN18` (and `python3 preprocess_wordnet.py -KBdataset WN18RR` )

  * If train/dev/test/ exists in `./data/WN**/`, you can use this preprocessor to another WN** datasets.

 * run `python3 preprocess_ent2type_desc_cano_reladj.py -entity_symbol2cano_type_desc_alreadydumped False -reverse_rel_data_alreadydumped False -KBdataset WN18 -spacy_model_str en_core_web_lg -multiprocess True`

### Preprocessing DBpedia**

* First Download
```
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/nif_context_en.ttl.bz2
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/infobox_properties_en.ttl.bz2
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/labels_en.ttl.bz2
```
to `./misc_data/dbpedia2016/`

then
`cd misc_data/dbpedia2016/`
`python dbpedia_ents_prep.py infobox_properties_en.ttl.bz2 labels_en.ttl.bz2 nif_context_en.ttl.bz2 > dbpedia_ents.text.jsonl`

Next, run context preprocessor.

```
python3 preprocess_dbpedia.py -jsonl2json_alreadydumped False -entitycheckalreadydone False -parsed_json_squeezing_by_entities_on_db50_and_db500_already False
nohup sh preprocess_dbpedia.sh > 191123_preprocess_dbpedia.log &
```
This will take 6-10 hours on 72 core cpus.