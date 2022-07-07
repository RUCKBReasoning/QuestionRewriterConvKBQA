

import random
import re
import os
import json
import difflib
import numpy as np
from tqdm import tqdm
from utils import load_sparql_retriever, formatAnswer, is_date, load_dict
from utils import ENTITY_PATTERN, RELATION_PATTERN, const_interaction_dic


def get_entity_relation(datasets, path_dict, sparql_retriever, elq_file, data_dir):
    entitiy_set, relation_set, relation_year_set = set(), set(), set()
    for dataset in datasets:
        all_data = json.load(open(os.path.join(dataset, elq_file)))
        for conv in tqdm(all_data):
            for q in conv['questions']:
                te_ids = q['te_ids'].split(";")
                answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
                answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
                answer_texts = [a.strip() for a in q['answer_text'].split(";")]
                if answers[0]=="n/a":answers=answer_texts.copy()
                if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
                for a in answers:entitiy_set.add(formatAnswer(a))

                for t in te_ids:
                    entitiy_set.add(formatAnswer(t))
                    paths = path_dict[t]
                    for p in paths:
                        if not re.match(RELATION_PATTERN, p[1]):continue
                        if is_date(formatAnswer(p[0])):
                            e = formatAnswer(p[0])
                            e = e.split(" ")[-1]
                            entitiy_set.add(e)
                            relation_year_set.add(p[1])
                        if is_date(formatAnswer(p[2])):
                            e = formatAnswer(p[2])
                            e = e.split(" ")[-1]
                            entitiy_set.add(e)
                            relation_year_set.add(p[1])
                                    
                        entitiy_set.add(formatAnswer(p[0]))
                        entitiy_set.add(formatAnswer(p[2]))
                        relation_set.add(p[1])
    with open(os.path.join(data_dir, "entities.txt"),"w") as f:
        for e in entitiy_set:
            f.write(e+"\n")
    with open(os.path.join(data_dir, "relations.txt"),"w") as f:
        for r in relation_set:
            if r in relation_year_set:
                f.write(sparql_retriever.wikidata_id_to_label(r)+" (year)\n") 
            f.write(sparql_retriever.wikidata_id_to_label(r)+"\n") 
        f.write("verify\n") # add yes/no relation
                     

def retrieve_subgraphs(dataset, entity2id, relation2id, path_dict, subgraph_dict, sparql_retriever, elq_file):
    new_data = []
    all_data = json.load(open(os.path.join(dataset, elq_file)))
    for conv in tqdm(all_data):
        for q in conv['questions']:
            line_dict = {}
            line_dict['id'] = q['question_id']
            line_dict['question'] = q['question']
            line_dict['entities'] = [entity2id[t] for t in q['te_ids'].split(";")]
            
            answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
            answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
            answer_texts = [a.strip() for a in q['answer_text'].split(";")]
            if answers[0]=="n/a":answers=answer_texts.copy()
            if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
            
            line_dict['answers'] = [{'kb_id':formatAnswer(a[0]),'text':a[1]} for a in zip(answers, answer_texts)]
            
            tuples, entity = set(), set()
            te_ids = q['te_ids'].split(";")
               
            for t in te_ids:
                paths = path_dict[t] 
                for p in paths:
                    if not re.match(RELATION_PATTERN, p[1]):continue
                    if is_date(formatAnswer(p[0])):
                        e = formatAnswer(p[0])
                        e = e.split(" ")[-1]
                        entity.add(entity2id[e])
                        entity.add(entity2id[formatAnswer(p[2])])
                        tuples.add((entity2id[e], relation2id[sparql_retriever.wikidata_id_to_label(p[1])+" (year)"], entity2id[formatAnswer(p[2])]))
                    if is_date(formatAnswer(p[2])):
                        e = formatAnswer(p[2])
                        e = e.split(" ")[-1]
                        entity.add(entity2id[e])
                        entity.add(entity2id[formatAnswer(p[0])])
                        tuples.add((entity2id[formatAnswer(p[0])], relation2id[sparql_retriever.wikidata_id_to_label(p[1])+" (year)"], entity2id[e]))          
                    
                    entity.add(entity2id[formatAnswer(p[0])])
                    entity.add(entity2id[formatAnswer(p[2])])
                    tuples.add((entity2id[formatAnswer(p[0])], relation2id[sparql_retriever.wikidata_id_to_label(p[1])], entity2id[formatAnswer(p[2])]))
                tuples.add((entity2id[t],len(relation2id.keys()),entity2id['yes'])) 
                tuples.add((entity2id[t],len(relation2id.keys()),entity2id['no'])) 
                entity.add(entity2id['yes']) 
                entity.add(entity2id['no'])
                
                if t not in subgraph_dict:
                    subgraph_dict[t] = {'tuples':list(tuples),'entities':list(entity)}

            line_dict['subgraph'] = {'tuples':list(tuples),'entities':list(entity)}
            new_data.append(line_dict)
    
    with open(os.path.join(dataset, dataset.split('/')[-1].split('_')[0]+"_simple.json"),"w") as f:
        for d in new_data:
            f.write(json.dumps(d) + "\n")


def retrieve_single_subgraph(single_data, entity2id, relation2id, path_dict, sparql_retriever, file_path):
    line_dict = {}
    line_dict['id'] = single_data['question_id']
    line_dict['question'] = single_data['raw_question']
    line_dict['entities'] = [entity2id[t] for t in single_data['te_ids']]
    line_dict['answers'] = [{'kb_id':formatAnswer(re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a[0])),'text':a[1]} for a in zip(re.sub('\.$', '', single_data['answer']).split(';'), single_data['answer_text'].split(";"))]

    tuples, entity = set(), set()
    te_ids = single_data['te_ids']
    for t in te_ids:
        paths = path_dict[t] 
        for p in paths:
            if not re.match(RELATION_PATTERN, p[1]):continue
            if is_date(formatAnswer(p[0])):
                e = formatAnswer(p[0])
                e = e.split(" ")[-1]
                entity.add(entity2id[e])
                entity.add(entity2id[formatAnswer(p[2])])
                tuples.add((entity2id[e], relation2id[sparql_retriever.wikidata_id_to_label(p[1])+" (year)"], entity2id[formatAnswer(p[2])]))
            if is_date(formatAnswer(p[2])):
                e = formatAnswer(p[2])
                e = e.split(" ")[-1]
                entity.add(entity2id[e])
                entity.add(entity2id[formatAnswer(p[0])])
                tuples.add((entity2id[formatAnswer(p[0])], relation2id[sparql_retriever.wikidata_id_to_label(p[1])+" (year)"], entity2id[e]))          
            entity.add(entity2id[formatAnswer(p[0])])
            entity.add(entity2id[formatAnswer(p[2])])
            tuples.add((entity2id[formatAnswer(p[0])], relation2id[sparql_retriever.wikidata_id_to_label(p[1])], entity2id[formatAnswer(p[2])]))
        tuples.add((entity2id[t],len(relation2id.keys()),entity2id['yes'])) 
        tuples.add((entity2id[t],len(relation2id.keys()),entity2id['no'])) 
        entity.add(entity2id['yes']) 
        entity.add(entity2id['no'])

    line_dict['subgraph'] = {'tuples':list(tuples),'entities':list(entity)}
    with open(file_path,"w") as f:
        f.write(json.dumps(line_dict) + "\n")
    
    cover = 0
    for a in line_dict['answers']:
        if entity2id[a['kb_id']] in line_dict['subgraph']['entities']:
            cover = 1
            break
    return cover


def generate_selftrain_data(datasets, entity2id, subgraphs, elq_file):
    for dataset in datasets:
        all_data = json.load(open(os.path.join(dataset, elq_file)))
        for c_idx,conv in tqdm(enumerate(all_data)):     
            for q_idx,q in enumerate(conv['questions']):
                answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
                answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
                answer_texts = [a.strip() for a in q['answer_text'].split(";")]
                if answers[0]=="n/a":answers=answer_texts.copy()
                if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
            
                answers = [formatAnswer(a) for a in answers]
                
                te_ids = q['te_ids'].split(";")
                te_texts = q['te_texts'].split(";")
                
                keep_or_not = 0
                for t_idx,t in enumerate(te_ids):
                    entities = subgraphs[t]['entities']
                    flag = 0
                    for a in answers:
                        if entity2id[a] in entities and te_texts[t_idx].lower() in q['rewrite'].lower():
                            keep_or_not = 1
                            flag = 1
                            break
                    if flag==1:break
                
                if keep_or_not == 1:
                    all_data[c_idx]['questions'][q_idx]['keep'] = 1
                else:
                    all_data[c_idx]['questions'][q_idx]['keep'] = 0
             
        json.dump(all_data, open(os.path.join(dataset, 'q_selftrain.json'),'w'),indent=4)


    

if __name__ == '__main__':
    
    sparql_dir = 'KB-cache/'
    sparql_retriever = load_sparql_retriever(sparql_dir)
    
    data_dir = 'Datasets/ConvQuestions'
    path_dict = json.load(open(os.path.join(data_dir, "paths_1hop.json")))
    datasets = [os.path.join(data_dir, dataset) for dataset in ['train_set', 'dev_set', 'test_set']]
    elq_file = 'rewrite_q_elq.json'
    get_entity_relation(datasets, path_dict, sparql_retriever, elq_file, data_dir)
    
    entity2id = load_dict(os.path.join(data_dir, "entities.txt"))
    relation2id = load_dict(os.path.join(data_dir, "relations.txt"))
    subgraph_dict = dict()
    for dataset in datasets:
        retrieve_subgraphs(dataset, entity2id, relation2id, path_dict, subgraph_dict, sparql_retriever, elq_file)
    json.dump(subgraph_dict, open(os.path.join(data_dir,'subgraph.json'), 'w'))

    subgraphs = json.load(open(os.path.join(data_dir, 'subgraph.json')))
    generate_selftrain_data(datasets, entity2id, subgraphs)
    
    
       