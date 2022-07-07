import os
import re
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
sys.path.append("../../BLINK/")
import elq.main_dense as main_dense

RELATION_PATTERN = re.compile('P[0-9\-]+')
ENTITY_PATTERN   = re.compile('Q[0-9]+')

class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()

def initELQ():
    print("init ELQ")
    models_path = "../../BLINK/models/" # the path where you stored the ELQ models
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


def getElqPredictions(args, models, id2wikidata, question_id, convquestion):
    data_to_link = [{"id": question_id, "text": convquestion}]
    predictions = main_dense.run(args, None, *models, test_data=data_to_link)
    elq_predictions = []
    for prediction in predictions:
        pred_scores = prediction["scores"]
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


def processSingleQuestion(args, models, id2wikidata, single_data, sparql_retriever, neighbor_dict, entity2id):
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
        with HiddenPrints():
            elq_predictions = getElqPredictions(args, models, id2wikidata, question_id, question) 
        te_ids = [t[0] for t in elq_predictions]
        te_texts = [sparql_retriever.wikidata_id_to_label(t) for t in te_ids]
        
        remove_idx = []
        for idx,t in enumerate(te_texts):
            if te_ids[idx] not in turn_neighbors or te_ids[idx] not in entity2id:
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
            if s_e_text not in question:
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
    
    # elq results   
    return question,te_ids,te_texts


def is_date(date):
	pattern = re.compile('^[0-9]+ [A-z]+ [0-9][0-9][0-9][0-9]$')
	if not(pattern.match(date.strip())):
		return False
	else:
		return True


def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    if not(pattern.match(timestamp)):
        return False
    else:
        return True


def convertTimestamp( timestamp):
    yearPattern = re.compile('^[0-9][0-9][0-9][0-9]-00-00T00:00:00Z')
    monthPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-00T00:00:00Z')
    dayPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    timesplits = timestamp.split("-")
    year = timesplits[0]
    if yearPattern.match(timestamp):
        return year
    month = convertMonth(timesplits[1])
    if monthPattern.match(timestamp):
        return month + " " + year
    elif dayPattern.match(timestamp):
        day = timesplits[2].rsplit("T")[0]
        return day + " " + month + " " +year
   
    return timestamp


def convertMonth( month):
    return{
        "01": "January",
        "02": "February",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "August",
        "09": "September", 
        "10": "October",
        "11": "November",
        "12": "December"
    }[month]


def formatAnswer(answer):
    if len(answer) == 0:
        return answer
    best_answer = answer
    if is_timestamp(answer):
        best_answer = convertTimestamp(answer)
    elif is_date(answer):
        day = answer.split(" ")[0]
        month = answer.split(" ")[1]
        year = answer.split(" ")[2]
        if len(day)==1:
            day = '0'+day 
        month = month.capitalize()
        best_answer = " ".join([day, month, year])
    elif answer == 'Yes' or answer == 'No':
        best_answer = answer.lower()
    elif ' (English)' in answer:
        best_answer = answer.replace(' (English)', '')
 
    return best_answer


def load_dict(filename):
    word2id = dict()
    with open(filename,"r") as f_in:
        for line in f_in:
            word = line.rstrip()
            word2id[word] = len(word2id)
    return word2id                        


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


def get_1hop_paths(te, sparql_retriever):
    statements, sparql_txts = sparql_retriever.SQL_1hop(((te,),))
    statements = hop_path_to_subgraph(statements)
    string_tails = sparql_retriever.SQL_string_entities(te)
    for r,t in string_tails:
        statements.append((te,r,t))
    return list(set(statements))


def te_text_in_q(te_texts, q):
    for te in te_texts:
        if te.lower() in q.lower():
            return True
    
    return False


def retrieve_single_subgraph(single_data, entity2id, relation2id, path_dict, sparql_retriever, file_path):
    line_dict = {}
    line_dict['id'] = single_data['question_id']
    line_dict['question'] = single_data['raw_question']
    line_dict['entities'] = [entity2id[t] for t in single_data['te_ids']]
    
    answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', single_data['answer']).split(';')]
    answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
    answer_texts = [a.strip() for a in single_data['answer_text'].split(";")]
    if answers[0]=="n/a":answers=answer_texts.copy()
    if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
    line_dict['answers'] = [{'kb_id':formatAnswer(a[0]),'text':a[1]} for a in zip(answers, answer_texts)]
    
    tuples, entity = set(), set()
    te_ids = single_data['te_ids']
    for t in te_ids:
        if t in path_dict:
            paths = path_dict[t] 
        else:
            paths = get_1hop_paths(t, sparql_retriever)
        for p in paths:
            if not re.match(RELATION_PATTERN, p[1]):continue
            if formatAnswer(p[0]) not in entity2id or formatAnswer(p[2]) not in entity2id or sparql_retriever.wikidata_id_to_label(p[1]) not in relation2id:continue
            if p[2]=='UNK':continue
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
    
    cover_ = 0
    for a in line_dict['answers']:
        if entity2id[a['kb_id']] in line_dict['subgraph']['entities'] and te_text_in_q(single_data['te_texts'], single_data['rewrite_question']):
            cover_ = 1
            break
    
    return cover, cover_



def get_single_pred_relations(e, question, relation_dict, path_dict, sparql_retriever, relation_retriever):
    q_r_pair = defaultdict(list)
    relations,paths = [], []
    if e in relation_dict:
        relations = relation_dict[e]
        paths = path_dict[e]
    else:
        statements, sparql_txts = sparql_retriever.SQL_1hop(((e,),))
        statements = hop_path_to_subgraph(statements)
        string_tails = sparql_retriever.SQL_string_entities(e)
        for r,t in string_tails:
            statements.append((e,r,t))
        paths = statements
        r = set()
        for p in statements:
            if re.match(RELATION_PATTERN, p[1]):
                r.add(sparql_retriever.wikidata_id_to_label(p[1]))
        relations = list(r)
        relation_dict[e] = relations
        path_dict[e] = statements
    
    k = e + " " + question           
    if len(relations)==0:return {k:[[],[],[],[],[]]}
    if len(relations)<5:relations=list(random.choices(relations,k=5))
    q_r_pair[k] = [[question, r] for r in relations]
    
    pred_rels = relation_retriever.infer_retriever(q_r_pair) 
    
    for k, rels in pred_rels.items():
        for idx, p_r in enumerate(rels):
            flag = 0
            for p in paths:
                r = p[1]
                if re.match(RELATION_PATTERN, r):
                    r = sparql_retriever.wikidata_id_to_label(r)
                if r==p_r:
                    t = formatAnswer(p[2])
                    if re.match(ENTITY_PATTERN, t):
                        t = sparql_retriever.wikidata_id_to_label(t)
                    pred_rels[k][idx] = [p_r, t]
                    flag=1
                    break
            assert flag==1
    return pred_rels


def process_CONVEX(fpath, is_pretrain=True, is_selftrain=False):
    cw_pairs = []
    with open(fpath, "r") as f:
        all_data = json.load(f)
        for conv in all_data:
            seed_entity = conv['seed_entity_text']
            last_context,last_answer = "", ""
            for idx,q in enumerate(conv['questions']):
                if idx == 0:
                    last_context = seed_entity+". "+q['question']+" " if q['question'].endswith("?") else seed_entity+". "+q['question']+"? "
                    last_answer = ", ".join(q['answer_text'].split(";"))+". "
                else:
                    last_context += last_answer + q['question']+" " if q['question'].endswith("?") else last_answer + q['question']+"? "
                    last_answer = ", ".join(q['answer_text'].split(";"))+". "
                
                if is_pretrain:
                    cw_pairs.append([last_context, ""])
                elif is_selftrain:
                    if q['keep'] == 1:
                        rewrite = q['rewrite']
                        cw_pairs.append([last_context, rewrite])
    return cw_pairs        


def process_CONVEX_rel(fpath, pred_rels, sep="[SEPE]", is_pretrain=False, is_selftrain=True):
    ENTITY_PATTERN = re.compile('Q[0-9]+')
    cw_pairs = []
    with open(fpath, "r") as f:
        all_data = json.load(f)
        for conv in all_data:
            s_e_id = re.match(ENTITY_PATTERN, re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', conv["seed_entity"])).group()
            seed_entity = conv['seed_entity_text']
            for idx,q in enumerate(conv['questions']):
                k = s_e_id+" "+q['question']
                context = seed_entity+", "+", ".join(pred_rels[k][0])+". "
                for i in range(0,idx):
                    question = conv['questions'][i]['question']
                    answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', conv['questions'][i]['answer']).split(';')]
                    answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
                    answer_texts = [a.strip() for a in conv['questions'][i]['answer_text'].split(";")]
                    context += question+" " if question.endswith("?") else question+"? "
                    for j,a in enumerate(answers):
                        k = a+" "+question
                        if re.match(ENTITY_PATTERN, a) and k in pred_rels:
                            context += answer_texts[j]+", "+", ".join(pred_rels[k][0])+". "
                        else:
                            context += answer_texts[j]+", "
                context += q['question']+" " if q['question'].endswith("?") else q['question']+"? "
                
                if is_selftrain:
                    rewrite = q['rewrite']
                    if q['keep'] == 1:
                        cw_pairs.append([context, rewrite])
                elif is_pretrain:
                    cw_pairs.append([context, ""])
    
    return cw_pairs