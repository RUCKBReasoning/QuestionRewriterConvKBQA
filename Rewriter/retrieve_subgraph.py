
import os
import re
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils import load_sparql_retriever, ENTITY_PATTERN, RELATION_PATTERN, const_interaction_dic


def get_topic_entity(data_dir, elq_file):
    topic_entities = set()
    topic_inter = set()
    for dataset in ['train_set', 'dev_set', 'test_set']:
        file_path = os.path.join(data_dir, dataset, elq_file)
        data = json.load(open(file_path))
        for conv in data:
            for q in conv['questions']:
                te_ids = set(q['te_ids'].split(";"))
                topic_entities.update(te_ids)
                
                answer_id = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in q['answer'].split(";")]
                answer_id = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answer_id]
                for a in answer_id:
                    if re.match(ENTITY_PATTERN, a):
                        topic_entities.add(a)
                        
                if len(te_ids) == 2 and re.search(const_interaction_dic, q['question']):
                    te_ids = tuple(sorted(te_ids))
                    const_type = tuple(re.findall('(?<= )%s(?= )' %const_interaction_dic, q['question']))
                    topic_inter.add((te_ids,const_type))         
    
    print('len of total topic entities: ', len(topic_entities))
    print('len of total topic interact: ', len(topic_inter))
    return topic_entities, topic_inter


def inter_path_to_subgraph(path_dict):
    subgraph = set()
    for key in path_dict:
        entities = path_dict[key]
        t1 = key[0][-1].split(":")[-1]
        t2 = key[1][-1].split(":")[-1]
        for e in entities:
            subgraph.add((e,key[0][-2],t1))
            subgraph.add((e,key[1][-2],t2))
    return list(subgraph)


def hop_path_to_subgraph(path_dict):
    subgraph = set()
    for key in path_dict:
        entities = path_dict[key]
        if len(entities) > 500:continue
        if len(key) == 1:#1hop
            idx = key[0].index('?e1')
            triple = list(key[0])
            for e in entities:
                triple[idx] = e
                subgraph.add(tuple(triple))
        if len(key) == 2:#2hop
            triple1 = list(key[0])
            triple2 = list(key[1])
            mid_key = ((key[0][0],key[0][1],'?e1'),)
            if mid_key in path_dict:
                mid_entities = path_dict[mid_key]
            else:
                continue
            for m in mid_entities:
                for e in entities:
                    triple1[2] = m
                    triple2[0] = m
                    triple2[2] = e
                    subgraph.add(tuple(triple1))
                    subgraph.add(tuple(triple2))
    return list(subgraph)   


def retrieve_2hop_paths(topic_entities, topic_inter, sparql_retriever, datadir="Datasets/ConvQuestions/"):
    raw_candidate_paths = defaultdict(list)
    for k in tqdm(topic_inter):
        topic_entity = k[0]
        const_type = k[1]
        key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in k])
        if const_type and (key not in sparql_retriever.STATEMENTS):
            statements, sparql_txts = sparql_retriever.SQL_1hop_interaction(((topic_entity[0],), (topic_entity[1],)), const_type)
            sparql_retriever.QUERY_TXT = sparql_retriever.QUERY_TXT.union(sparql_txts)
            sparql_retriever.STATEMENTS[key].update(statements)
        else:
            statements = sparql_retriever.STATEMENTS[key]
        if statements:
            statements = inter_path_to_subgraph(statements)
            raw_candidate_paths[key] = statements
    
    for te in tqdm(topic_entities):
        key = (te, None)
        key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
        if key not in sparql_retriever.STATEMENTS:
            statements, sparql_txts = sparql_retriever.SQL_1hop(((te,),), sparql_retriever.QUERY_TXT)
            sparql_retriever.QUERY_TXT = sparql_retriever.QUERY_TXT.union(sparql_txts)
            num_1hop = sum([len(tails) for tails in statements.values()])
            if num_1hop < 5000:
                statements_tmp, sparql_txts = sparql_retriever.SQL_2hop(((te,),), sparql_retriever.QUERY_TXT)
                sparql_retriever.QUERY_TXT = sparql_retriever.QUERY_TXT.union(sparql_txts)
                statements.update(statements_tmp)
            sparql_retriever.STATEMENTS[key].update(statements)
        else:
            statements = sparql_retriever.STATEMENTS[key]
        if statements:
            statements = hop_path_to_subgraph(statements)
            raw_candidate_paths[te] = statements
    
    json.dump(raw_candidate_paths, open(os.path.join(datadir,'paths_2hop.json'),'w'),indent=4)      


def retrieve_1hop_paths(topic_entities, sparql_retriever, path_dict=None, datadir="Datasets/ConvQuestions/"):   
    if path_dict:
        raw_candidate_paths = path_dict
    else:
        raw_candidate_paths = defaultdict(list)
    print(len(raw_candidate_paths.keys()))
    for te in tqdm(topic_entities):
        if te in raw_candidate_paths:continue
        
        statements, sparql_txts = sparql_retriever.SQL_1hop(((te,),))
        statements = hop_path_to_subgraph(statements)
        string_tails = sparql_retriever.SQL_string_entities(te)
        for r,t in string_tails:
            statements.append((te,r,t))
        raw_candidate_paths[te] = list(set(statements))
    print(len(raw_candidate_paths.keys()))
    json.dump(raw_candidate_paths, open(os.path.join(datadir, 'paths_1hop.json'),'w'),indent=4) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ELQ
    parser.add_argument("--pre_train",            action='store_true')
    parser.add_argument("--self_train",           action='store_true')
    args = parser.parse_args()
    if args.pre_train:
        elq_file = 'rewrite_q_elq.json'
    elif args.self_train:
        elq_file = 'rewrite_q_selftrain_rr_elq.json'
    else:
        print("Please input the argument ( pre_train or self_train).")

    data_dir = 'Datasets/ConvQuestions'
    sparql_dir = 'KB-cache/'
    sparql_retriever = load_sparql_retriever(sparql_dir)
    topic_entities, topic_inter = get_topic_entity(data_dir, elq_file)
    retrieve_1hop_paths(topic_entities, sparql_retriever, datadir=data_dir)
    #retrieve_2hop_paths(topic_entities, topic_inter, sparql_retriever, data_dir)
    
    
    