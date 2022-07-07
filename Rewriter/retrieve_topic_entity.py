
import os
import re
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils import load_sparql_retriever, HiddenPrints, ENTITY_PATTERN, formatAnswer

sys.path.append("BLINK/")
import elq.main_dense as main_dense

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def initELQ():
    #data+config relevant for ELQ NED
    print("init ELQ")
    models_path = "BLINK/models/" # the path where you stored the ELQ models
    config = {
        "interactive": False,
        "biencoder_model": models_path+"elq_wiki_large.bin",
        "biencoder_config": models_path+"elq_large_params.txt",
        "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
        "entity_catalogue": models_path+"entity.jsonl",
        "entity_encoding": models_path+"all_entities_large.t7",
        "output_path": "logs/", # logging directory
        "faiss_index": "hnsw",
        "index_path": models_path+"faiss_hnsw_index.pkl",
        "num_cand_mentions": 10,
        "num_cand_entities": 10,
        "threshold_type": "joint",
        "threshold": -4.5,
    }
    args = argparse.Namespace(**config)
    models = main_dense.load_models(args, logger=None)
    id2wikidata = json.load(open("models/id2wikidata.json"))
    print("init done")
    return args, models, id2wikidata


def get_seed_entity(data_dir):
    seed_entities = set()
    for dataset in ['train_set', 'dev_set', 'test_set']:
        file_path = os.path.join(data_dir, dataset, 'q.json')
        data = json.load(open(file_path))
        for conv in data:
            seed_id = re.match(ENTITY_PATTERN, re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', conv["seed_entity"])).group()
            seed_entities.add(seed_id)
    print('len of seed entities: ', len(seed_entities))
    return seed_entities


def get_seed_neighbors(data_dir, seed_entities, sparql_retriever):
    neighbors = defaultdict(list)
    for seed in seed_entities:
        key = (seed, None)
        key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
        if key not in sparql_retriever.STATEMENTS:
            statements, sparql_txts = sparql_retriever.SQL_1hop(((seed,),), sparql_retriever.QUERY_TXT)
            sparql_retriever.QUERY_TXT = sparql_retriever.QUERY_TXT.union(sparql_txts)
            sparql_retriever.STATEMENTS[key].update(statements)
        else:
            statements = sparql_retriever.STATEMENTS[key]
        if statements:
            tails = set([seed]) # add seed itself
            for key in statements:
                tails.update(statements[key])
            neighbors[seed] = list(tails)
    json.dump(neighbors, open(os.path.join(data_dir, "seed_neighbors.json"),'w'))


def get_entities(data_dir):
    all_entities = set()
    for dataset in ['train_set', 'dev_set', 'test_set']:
        file_path = os.path.join(data_dir, dataset, 'q.json')
        data = json.load(open(file_path))
        for conv in data:
            seed_id = re.match(ENTITY_PATTERN, re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', conv["seed_entity"])).group()
            all_entities.add(seed_id)
            for q in conv['questions']:
                answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
                answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
                answers = [formatAnswer(a) for a in answers]
                answer_texts = [a.strip() for a in q['answer_text'].split(";")]
                if answers[0]=="n/a":answers=answer_texts.copy()
                all_entities.update(set(answers))
    print('len of all entities: ', len(all_entities))
    return all_entities


def get_entity_neighbors(data_dir, all_entities, sparql_retriever):
    neighbors = defaultdict(list)
    for entity in all_entities:
        if not re.match(ENTITY_PATTERN, entity):
            neighbors[entity] = [entity]
            continue
        statements, sparql_txts = sparql_retriever.SQL_1hop(((entity,),))
        if statements:
            tails = set([entity]) # add entity itself
            for key in statements:
                tails.update(statements[key])
            neighbors[entity] = list(tails)
    json.dump(neighbors, open(os.path.join(data_dir, "neighbors.json"),'w'), indent=4)


def get_single_entity_neighbors(entity, sparql_retriever):
    neighbors = []
    if not re.match(ENTITY_PATTERN, entity):
        neighbors = [entity]
    else:    
        statements, sparql_txts = sparql_retriever.SQL_1hop(((entity,),))
        if statements:
            tails = set([entity]) # add entity itself
            for key in statements:
                tails.update(statements[key])
            neighbors = list(tails)
    return neighbors    
    

def getElqPredictions(args, models, id2wikidata, question_id, convquestion):
    data_to_link = [{"id": question_id, "text": convquestion}]
    #run elq to get predictions for current conversational question
    predictions = main_dense.run(args, None, *models, test_data=data_to_link)
    elq_predictions = []
    for prediction in predictions:
        pred_scores = prediction["scores"]
        #get entity ids from wikidata
        pred_ids = [id2wikidata.get(wikipedia_id) for (wikipedia_id, a, b) in prediction['pred_triples']]
        p=0
        for predId in pred_ids:
            if predId is None:
                continue
            #normalize the score
            score = np.exp(pred_scores[p])
            i = 0
            modified = False
            #potentially update score if same entity is matched multiple times
            for tup in elq_predictions:
                if tup[0] == predId:
                    modified = True   
                    if score > tup[1]:
                        elq_predictions[i] = (predId, score)
                i += 1
            #store enitity id along its normalized score
            if not modified:
                elq_predictions.append((predId, score))
            p+=1
    
    return elq_predictions


def checkRWQuestion(raw_question, rewrite_question, entities):
    if "?" in rewrite_question:
        if rewrite_question.endswith("?"):   
            rewrite_question = rewrite_question.split("?")[-2].strip()+"?"      
        else:
            rewrite_question = rewrite_question.split("?")[-1].strip()+"?" 
            
    tag = " ".join(rewrite_question.split(" ")[-2:])
    for e in entities:
        raw_idx = e.lower().rfind(tag.lower())
        if raw_idx!=-1:
            concat = e
            break
    if raw_idx==-1:
        raw_idx = raw_question.rfind(tag)    
        concat = raw_question     
    re_idx = rewrite_question.rfind(tag)
    
    if raw_idx==-1:
        check_question = rewrite_question
    else:
        check_question = rewrite_question[:re_idx]+concat[raw_idx:]
    if not check_question.endswith("?"):
        check_question += "?"
    
    return check_question   


def processData(args, models, sparql_retriever, id2wikidata, neighbors, data, data_dir, dataset, elq_file):
    for conv in tqdm(data):
        s_e_id = re.match(ENTITY_PATTERN, re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', conv["seed_entity"])).group()
        s_e_text = conv["seed_entity_text"]
        entities = {s_e_id:s_e_text}
        #go over each question
        for question_info in conv['questions']:
            question_id = question_info['question_id']         
            #raw_question = question_info['question']
            question = question_info["rewrite"]
            #question = checkRWQuestion(raw_question, rewrite_question, list(entities.values()))
            turn_neighbors = set()
            for k in entities:
                if k in neighbors:
                    turn_neighbors.update(set(neighbors[k]))
                else:
                    ns = get_single_entity_neighbors(k, sparql_retriever)
                    turn_neighbors.update(set(ns))
                    neighbors[k] = ns
            
            while True:
                #get predictions from NED tool as one scoring factor, include conv. history for better results
                with HiddenPrints():
                    elq_predictions = getElqPredictions(args, models, id2wikidata, question_id, question) 
                te_ids = [t[0] for t in elq_predictions]
                te_texts = [sparql_retriever.wikidata_id_to_label(t) for t in te_ids]
                
                remove_idx = []
                # check elq results
                for idx,t in enumerate(te_texts):
                    if te_ids[idx] not in turn_neighbors:
                        remove_idx.append(idx)
                        continue
                    if t in entities.values():#check id 
                        flag = 0
                        for k in entities:
                            if entities[k]==t and k==te_ids[idx]:flag=1
                        if flag==0:
                            for k in entities:
                                if entities[k]==t:
                                    te_ids[idx] = k
                te_ids = np.delete(te_ids, remove_idx).tolist()
                te_texts = np.delete(te_texts, remove_idx).tolist()
                
                #check overlap of text of entity name  
                for eid,et in entities.items():
                    if eid not in te_ids and et in question:
                        te_ids.append(eid)
                        te_texts.append(et)
                        
                assert len(te_ids) == len(te_texts)  
                                   
                if len(te_texts)==0:
                    if s_e_text not in question:#empty
                        question = s_e_text + ", " + question
                        continue
                    else:
                        te_ids = [s_e_id]
                        te_texts = [s_e_text]
                        break
                elif len(te_texts)==0 and s_e_text + ", " in question:
                    te_ids = [s_e_id]
                    te_texts = [s_e_text]
                    question = question.replace(s_e_text + ", ", "", 1)
                    break
                else:
                    if s_e_text + ", " in question:
                        question = question.replace(s_e_text + ", ", "", 1)
                    break        
            
            answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', question_info['answer']).split(';')]
            answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
            answers = [formatAnswer(a) for a in answers]
            answer_texts = [a.strip() for a in question_info['answer_text'].split(";")]
            if answers[0]=="n/a":answers=answer_texts.copy()
            if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
            es = answer_texts + te_texts
            ids = answers + te_ids
            
            for idx,id in enumerate(ids):
                if id not in entities and re.match(ENTITY_PATTERN, id):
                    entities[id] = es[idx]    
            
            # store elq results   
            #question_info["rewrite"] = question
            question_info['te_ids'] = ";".join(te_ids)
            question_info['te_texts'] = ";".join(te_texts)
            
    json.dump(data, open(os.path.join(data_dir, dataset, elq_file), "w"), indent=4)
    json.dump(neighbors, open(os.path.join(data_dir, "neighbors.json"),'w'), indent=4)    
    
    return


def processSingleQuestion(args, models, id2wikidata, single_data, sparql_retriever, neighbor_dict):
    question_id      = single_data['question_id']
    raw_question     = single_data['raw_question']
    rewrite_question = single_data['rewrite_question']
    entities         = single_data['entities']
    s_e_id           = single_data['s_e_id']
    s_e_text         = single_data['s_e_text']
    
    #question = checkRWQuestion(raw_question, rewrite_question, list(entities.values()))
    question = rewrite_question
    
    turn_neighbors = set()
    for k in entities:
        if k in neighbor_dict:
            turn_neighbors.update(set(neighbor_dict[k]))
        else:
            ns = get_single_entity_neighbors(k, sparql_retriever)
            turn_neighbors.update(set(ns))
            neighbor_dict[k] = ns
    turn_neighbors = neighbor_dict[s_e_id]
    
    while True:
        #get predictions from NED tool as one scoring factor, include conv. history for better results
        with HiddenPrints():
            elq_predictions = getElqPredictions(args, models, id2wikidata, question_id, question) 
        te_ids = [t[0] for t in elq_predictions]
        te_texts = [sparql_retriever.wikidata_id_to_label(t) for t in te_ids]
        
        remove_idx = []
        # check elq results
        for idx,t in enumerate(te_texts):
            if te_ids[idx] not in turn_neighbors:
                remove_idx.append(idx)
                continue
            if t in entities.values():#check id 
                flag = 0
                for k in entities:
                    if entities[k]==t and k==te_ids[idx]:flag=1
                if flag==0:
                    for k in entities:
                        if entities[k]==t:
                            te_ids[idx] = k
        te_ids = np.delete(te_ids, remove_idx).tolist()
        te_texts = np.delete(te_texts, remove_idx).tolist()
        
        assert len(te_ids) == len(te_texts) 
        
        if len(te_texts)==0:
            if s_e_text not in question:#empty
                question = s_e_text + ", " + question
                continue
            else:
                te_ids = [s_e_id]
                te_texts = [s_e_text]
                break
        elif len(te_texts)==0 and s_e_text in question:
            te_ids = [s_e_id]
            te_texts = [s_e_text]
            if s_e_text + ", " in question:
                question = question.replace(s_e_text + ", ", "", 1)
            break
        else:
            if s_e_text + ", " in question:
                question = question.replace(s_e_text + ", ", "", 1)
            break            
       
    return question,te_ids,te_texts

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ELQ
    parser.add_argument("--pre_train",            action='store_true')
    parser.add_argument("--self_train",           action='store_true')
    args = parser.parse_args()
    if args.pre_train:
        question_file = 'rewrite_q.json'
        elq_file = 'rewrite_q_elq.json'
    elif args.self_train:
        question_file = 'rewrite_q_selftrain_rr.json'
        elq_file = 'rewrite_q_selftrain_rr_elq.json'
    else:
        print("Please input the argument ( pre_train or self_train).")
    
    sparql_dir = 'KB-cache/'
    sparql_retriever = load_sparql_retriever(sparql_dir)
    
    data_dir = 'Datasets/ConvQuestions'
    seed_entities = get_seed_entity(data_dir)
    get_seed_neighbors(data_dir, seed_entities, sparql_retriever)
    all_entities = get_entities(data_dir)
    get_entity_neighbors(data_dir, all_entities, sparql_retriever)
    
    neighbors = json.load(open(os.path.join(data_dir,"neighbors.json")))
    args, models, id2wikidata = initELQ()
    for dataset in ['train_set', 'dev_set', 'test_set']:
        with open(os.path.join(data_dir, dataset, question_file)) as qwFile:
            qws_data = json.load(qwFile)
            processData(args, models, sparql_retriever, id2wikidata, neighbors, qws_data, data_dir, dataset, elq_file) 
            