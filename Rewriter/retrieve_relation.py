

import os
import random
import re
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from retrieve_subgraph import retrieve_1hop_paths
from utils import load_sparql_retriever, formatAnswer, HiddenPrints
from utils import ENTITY_PATTERN, RELATION_PATTERN
from train_relation_retriever import RelationRetriever
from train_rewriter import CQR
from retrieve_topic_entity import initELQ, processSingleQuestion
from retrieve_subgraph import hop_path_to_subgraph



def get_1hop_relations(data_dir, path_dict, sparql_retriever):
    entities = set()
    for dataset in ['train_set', 'dev_set', 'test_set']:
        fpath = os.path.join(data_dir, dataset, 'q.json')
        all_data = json.load(open(fpath))
        for conv in all_data:
            for q in conv['questions']:
                answer_id = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in q['answer'].split(";")]
                answer_id = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answer_id]
                for a in answer_id:
                    if re.match(ENTITY_PATTERN, a) and a not in path_dict:
                        entities.add(a)
    # print(len(entities))
    # print(len(path_dict.keys()))
    retrieve_1hop_paths(list(entities), sparql_retriever, path_dict)


def process_1hop_relations(data_dir, path_dict, sparql_retriever):
    # print(len(path_dict.keys()))
    relations = defaultdict(list)
    for e, paths in tqdm(path_dict.items()):
        r = set()
        for p in paths:
            if re.match(RELATION_PATTERN, p[1]):
                r.add(sparql_retriever.wikidata_id_to_label(p[1]))
        relations[e] = list(r)
    assert len(relations.keys()) == len(path_dict.keys())
    json.dump(relations, open(os.path.join(data_dir, 'relations_1hop.json'), 'w'), indent=4)


def generate_relation_dataset(data_dir, relation_dict, path_dict, sparql_retriever):
    for dataset in ['train_set', 'dev_set', 'test_set']:
        relation_data = []
        question_set = set()
        fpath = os.path.join(data_dir, dataset, 'q.json')
        all_data = json.load(open(fpath))
        for conv in tqdm(all_data):
            s_e_id = re.match(ENTITY_PATTERN, re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', conv["seed_entity"])).group()
            entities = [s_e_id]
            for q in conv['questions']:
                q_data = {'question':q['question'], 'pos_r':[], 'neg_r':[]}
                answer_id = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in q['answer'].split(";")]
                answer_id = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answer_id]
                answer_id = [formatAnswer(a) for a in answer_id]
                
                for e in entities:
                    paths = path_dict[e]
                    for a in answer_id:
                        for p in paths:
                            if (formatAnswer(p[2])==a) and re.match(RELATION_PATTERN, p[1]):
                                q_data['pos_r'].append(sparql_retriever.wikidata_id_to_label(p[1]))
                    q_data['pos_r'] = list(set(q_data['pos_r']))
                    q_data['neg_r'] = list(set(relation_dict[e]) - set(q_data['pos_r']))
                    if len(q_data['pos_r']) > 0 and len(q_data['neg_r']) > 0:
                        relation_data.append(q_data)
                        question_set.add(q_data['question'])
                        
                        break
                for a in answer_id:
                    if re.match(ENTITY_PATTERN, a) and a not in entities:entities.append(a)
        print(len(relation_data))
        with open(os.path.join(data_dir, dataset, dataset.split('_')[0]+"_relation.json"),"w") as f:
            for d in relation_data:
                f.write(json.dumps(d) + "\n")           


def get_pred_relations(data_dir, relation_dict, path_dict, sparql_retriever):
    all_q_r_pair = defaultdict(list)
    for dataset in ['train_set', 'dev_set', 'test_set']:
        for filename in ['rewrite_q_elq.json', 'rewrite_q_selftrain_elq.json', 'rewrite_q_selftrain_rr_elq.json']:
            fpath = os.path.join(data_dir, dataset, filename)
            all_data = json.load(open(fpath))
            for conv in all_data:
                s_e_id = re.match(ENTITY_PATTERN, re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', conv["seed_entity"])).group()
                last_e = [s_e_id]
                for q in conv['questions']:
                    te_ids = q['te_ids'].split(";")
                    for t in te_ids:
                        if re.match(ENTITY_PATTERN, t) and t not in last_e:last_e.append(t)
                    
                    question = q['question']
                    for e in last_e:
                        relations = relation_dict[e]
                        if len(relations)==0:continue
                        if len(relations)<5:relations=list(random.choices(relations,k=5))
                        k = e + " " + question
                        if k not in all_q_r_pair:
                            all_q_r_pair[k] = [[question, r] for r in relations]
                    
                    #last_e.clear()
                    answer_id = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in q['answer'].split(";")]
                    answer_id = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answer_id]
                    for a in answer_id:
                        if re.match(ENTITY_PATTERN, a) and a not in last_e:last_e.append(a)
    print(len(all_q_r_pair))
    
    config_fn = 'config/config_relation_retriever.json'
    with open(config_fn) as f:
        args = json.load(f)
    relation_retriever =  RelationRetriever(args)
    pred_rels = relation_retriever.infer_retriever(all_q_r_pair) 
    
    for k, rels in tqdm(pred_rels.items()):
        e  = k.split(" ")[0]
        paths = path_dict[e]
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
    json.dump(pred_rels, open(os.path.join(data_dir,'pred_relations.json'), "w"), indent=4) 
                   

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


def isExistential(question_start):
    existential_keywords = ['is', 'are', 'was', 'were', 'am', 'be', 'being', 'been', 'did', 'do', 'does', 'done', 'doing', 'has', 'have', 'had', 'having']
    if question_start in existential_keywords:
        return True
    return False
      

def test_rr_final():
    config_fn = 'config/config_relation_retriever.json'
    print("Loading relation retriever model from {}".format(config_fn))
    with open(config_fn) as f:
        args = json.load(f)
    relation_retriever =  RelationRetriever(args)
    relation_retriever.load_model()
    
    config_fn = 'config/config_ConvQuestions_selftrain.json'
    print("Loading finetuned t5 model from {}".format(config_fn))
    with open(config_fn) as f:
        t5_args = json.load(f)
    cqr = CQR(t5_args)  
    
    elq_args, models, id2wikidata = initELQ()
    sparql_dir = 'KB-cache'
    sparql_retriever = load_sparql_retriever(sparql_dir)
    
    
    path_dict     = json.load(open("Datasets/ConvQuestions/paths_1hop.json"))
    neighbor_dict = json.load(open("Datasets/ConvQuestions/neighbors.json"))
    relation_dict = json.load(open("Datasets/ConvQuestions/relations_1hop.json"))
    pred_rel_dict = json.load(open("Datasets/ConvQuestions/pred_relations.json"))
    
    ENTITY_PATTERN = re.compile('Q[0-9]+')
    
    eval_path = 'Datasets/ConvQuestions/test_set/q.json'
    all_test_data = json.load(open(eval_path)) 
    total,hit1,hit3,hit5, total_f1 = 0,0,0,0,0.0
    total_mo,hit1_mo,hit3_mo,hit5_mo, total_f1_mo = 0,0,0,0,0.0
    total_mu,hit1_mu,hit3_mu,hit5_mu, total_f1_mu = 0,0,0,0,0.0
    total_tv,hit1_tv,hit3_tv,hit5_tv, total_f1_tv = 0,0,0,0,0.0
    total_so,hit1_so,hit3_mo,hit5_so, total_f1_so = 0,0,0,0,0.0
    total_bo,hit1_bo,hit3_mo,hit5_bo, total_f1_bo = 0,0,0,0,0.0
    
    for conv in tqdm(all_test_data):
        s_e_id = re.match(ENTITY_PATTERN, re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', conv["seed_entity"])).group()
        s_e_text = conv["seed_entity_text"]
        domain = conv["domain"]
        entities = {s_e_id:s_e_text}
        
        last_context,last_answer = "", ""
        last_answer_ids,last_answer_texts = [],[]
        for idx,q in enumerate(conv['questions']):
            answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
            answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
            answer_texts = [a.strip() for a in q['answer_text'].split(";")]
            if answers[0]=="n/a":answers=answer_texts.copy()
            if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
        
            answers = [formatAnswer(a) for a in answers]
            #print(answers)
            
            if domain=="movies":
                total_mo += 1
            elif domain=="tv_series" :
                total_tv += 1
            elif domain=="music":
                total_mu += 1
            elif domain=="books":
                total_bo += 1
            elif domain=="soccer":
                total_so += 1
            
            
            if isExistential(q['question'].split(" ")[0].lower()):
                a = 'yes'
                if a in answers:
                    flag = 1
                    if idx<1:
                        hit1 += 1
                        correct = 1
                        retrieved=[a,'no']
                        p, r = correct / len(retrieved), correct / len(answers)
                        f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
                        total_f1 += f1
                        
                        if domain=="movies":
                            hit1_mo += 1
                            total_f1_mo += f1
                        elif domain=="tv_series" :
                            hit1_tv += 1
                            total_f1_tv += f1
                        elif domain=="music":
                            hit1_mu += 1
                            total_f1_mu += f1
                        elif domain=="books":
                            hit1_bo += 1
                            total_f1_bo += f1
                        elif domain=="soccer":
                            hit1_so += 1
                            total_f1_so += f1
                            
                        
                        #break
                    if idx<3:
                        hit3 += 1
                        #break
                    if idx<5:
                        hit5 += 1
                total += 1
            else:
                # if idx == 0:
                #     last_context = s_e_text +". "+q['question']+" " if q['question'].endswith("?") else s_e_text +". "+q['question']+"? "
                # else:
                #     last_context += last_answer + q['question']+" " if q['question'].endswith("?") else last_answer + q['question']+"? "
                    

                k = s_e_id+" "+q['question']
                context = s_e_text+", "+", ".join(pred_rel_dict[k][0])+". "
                for i in range(0,idx):
                    question = conv['questions'][i]['question']
                    pred_answers = last_answer_ids
                    pred_answer_texts = last_answer_texts
                    context += question+" " if question.endswith("?") else question+"? "
                    for j,a in enumerate(pred_answers):
                        k = a+" "+q['question']
                        if re.match(ENTITY_PATTERN, a) and k in pred_rel_dict:
                            context += pred_answer_texts[j]+", "+", ".join(pred_rel_dict[k][0])+". "
                        elif re.match(ENTITY_PATTERN, a) and k not in pred_rel_dict:
                            single_pred_rels = get_single_pred_relations(a, q['question'],relation_dict,path_dict,sparql_retriever, relation_retriever)
                            context += pred_answer_texts[j]+", "+", ".join(single_pred_rels[k][0])+". "
                        else:
                            context += pred_answer_texts[j]+", "
                context += q['question']+" " if q['question'].endswith("?") else q['question']+"? "
                
                
                question = cqr.rewrite_single_question(context=context)
                single_data = {'question_id':      q['question_id'],
                                'raw_question':     q['question'],
                                'rewrite_question': question,
                                'entities':         entities,
                                's_e_id':           s_e_id,
                                's_e_text':         s_e_text,
                                'answer':           q['answer'],
                                'answer_text':      q['answer_text']
                                }
                with HiddenPrints():
                    check_question,te_ids,te_texts = processSingleQuestion(elq_args, models, id2wikidata, single_data, sparql_retriever, neighbor_dict)
                single_data['rewrite_question'] = check_question
                single_data['te_ids']           = te_ids
                single_data['te_texts']         = te_texts
                
                
                for te in te_ids:
                    flag = 0
                    
                    relations, paths = [], []
                    if te in relation_dict:
                        relations = relation_dict[te]
                        paths = path_dict[te]
                    else:
                        statements, sparql_txts = sparql_retriever.SQL_1hop(((te,),))
                        statements = hop_path_to_subgraph(statements)
                        string_tails = sparql_retriever.SQL_string_entities(te)
                        for r,t in string_tails:
                            statements.append((te,r,t))
                        paths = statements
                        r = set()
                        for p in statements:
                            if re.match(RELATION_PATTERN, p[1]):
                                r.add(sparql_retriever.wikidata_id_to_label(p[1]))
                        relations = list(r)
                        relation_dict[te] = relations
                        path_dict[te] = statements
                        
                    #relations = relation_dict[te]
                    if len(relations)==0:continue
                    if len(relations)<5:relations=list(random.choices(relations,k=5))
                    
                    k = te + " " + single_data['raw_question']
                    pred_rels = relation_retriever.infer_retriever({k:[[single_data['raw_question'], r] for r in relations]}) 

                    for k, rels in pred_rels.items():
                        # e  = k.split(" ")[0]
                        # paths = path_dict[e]
                        for idx, p_r in enumerate(rels):
                            for p in paths:
                                r = p[1]
                                if re.match(RELATION_PATTERN, r):
                                    r = sparql_retriever.wikidata_id_to_label(r)
                                if r==p_r:
                                    a = formatAnswer(p[2])
                                    break
                            #print(a)
                            
                            if a in answers:
                                flag = 1
                                if idx<1:
                                    hit1 += 1
                                    correct = 1
                                    retrieved=[a]
                                    p, r = correct / len(retrieved), correct / len(answers)
                                    f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
                                    total_f1 += f1
                                    
                                    if domain=="movies":
                                        hit1_mo += 1
                                        total_f1_mo += f1
                                    elif domain=="tv_series" :
                                        hit1_tv += 1
                                        total_f1_tv += f1
                                    elif domain=="music":
                                        hit1_mu += 1
                                        total_f1_mu += f1
                                    elif domain=="books":
                                        hit1_bo += 1
                                        total_f1_bo += f1
                                    elif domain=="soccer":
                                        hit1_so += 1
                                        total_f1_so += f1
                                    
                                    #break
                                if idx<3:
                                    hit3 += 1
                                    #break
                                if idx<5:
                                    hit5 += 1
                                    #break
                                break
                    if flag==1:break
                total += 1
            
            if re.match(ENTITY_PATTERN, a):
                nsm_answer_text = sparql_retriever.wikidata_id_to_label(a)
            else:
                nsm_answer_text = a
            
            # update entities, ner+answer in last turn
            es = [nsm_answer_text] + te_texts
            ids = [a] + te_ids
            for idx,id in enumerate(ids):
                if id not in entities and re.match(ENTITY_PATTERN, id):
                    entities[id] = es[idx]  
            
            last_answer = nsm_answer_text +". "
            last_answer_ids.append(a)
            last_answer_texts.append(nsm_answer_text)
            #break
    
    print("h1: ", hit1/total, total)
    print("h3: ", hit3/total, total)
    print("h5: ", hit5/total, total)
    print("f1: ", total_f1/total)
    
    print("movies h1: ", hit1_mo/total_mo, total_mo)
    print("movies f1: ", total_f1_mo/total_mo)
    print("tv series h1: ", hit1_tv/total_tv, total_tv)
    print("tv series f1: ", total_f1_tv/total_tv)
    print("music h1: ", hit1_mu/total_mu, total_mu)
    print("music f1: ", total_f1_mu/total_mu)
    print("soccer h1: ", hit1_so/total_so, total_so)
    print("soccer f1: ", total_f1_so/total_so)
    print("books h1: ", hit1_bo/total_bo, total_bo)
    print("books f1: ", total_f1_bo/total_bo)



if __name__ == '__main__':
    data_dir = 'Datasets/ConvQuestions'
    path_dict = json.load(open(os.path.join(data_dir, 'paths_1hop.json')))
    
    sparql_dir = 'KB-cache/'
    sparql_retriever = load_sparql_retriever(sparql_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--construct",            action='store_true')
    parser.add_argument("--infer",                action='store_true')
    parser.add_argument("--eval",                 action='store_true')
    args = parser.parse_args()
    if args.construct:
        get_1hop_relations(data_dir, path_dict, sparql_retriever)
        process_1hop_relations(data_dir, path_dict, sparql_retriever)
        relation_dict = json.load(open(os.path.join(data_dir, 'relations_1hop.json')))
        generate_relation_dataset(data_dir, relation_dict, path_dict, sparql_retriever)
    elif args.infer:
        relation_dict = json.load(open(os.path.join(data_dir, 'relations_1hop.json')))
        get_pred_relations(data_dir, relation_dict, path_dict, sparql_retriever)
    elif args.eval:
        test_rr_final()
    else:
        print("Please input the argument ( construct, infer or eval).")
