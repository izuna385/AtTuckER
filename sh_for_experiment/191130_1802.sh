CUDA_VISIBLE_DEVICES=0  python3 main.py -KBdataset dbpedia50 -training_dataset_augument_by_multisent False -num_epochs 500 -relation_attention2headandtaildef False -batch_size 1024 -definition_seq2seq multiheadstackatt  -embedding_strategy pretrained -encoder bow -only_consider_its_one_gold True