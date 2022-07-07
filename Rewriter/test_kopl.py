import re
import os
import sys
import json
from tqdm import tqdm

from train_rewriter import CQR
from train_relation_retriever import RelationRetriever
from retrieve_topic_entity import initELQ, processSingleQuestion
from utils import load_sparql_retriever, load_dict, HiddenPrints
from retrieve_relation import get_single_pred_relations

sys.path.append('Reasoner/KoPL/code')
from infer_ConvQuestions import test_kopl, formatAnswer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def test_final_h1():
    
    config_fn = 'config/config_ConvQuestions_selftrain.json'
    print("Loading finetuned t5 model from {}".format(config_fn))
    with open(config_fn) as f:
        t5_args = json.load(f)
    cqr = CQR(t5_args)  
    
    config_fn = 'config/config_relation_retriever.json'
    print("Loading relation retriever model from {}".format(config_fn))
    with open(config_fn) as f:
        rr_args = json.load(f)
    relation_retriever = RelationRetriever(rr_args)
    
    sparql_dir = 'KB-cache'
    sparql_retriever = load_sparql_retriever(sparql_dir)
    
    elq_args, models, id2wikidata = initELQ()
    
    id_to_name = json.load(open("Reasoner/KoPL/KB/id_to_name.json"))
    name_to_id = {name:idx for idx, name in id_to_name.items()}
    ENTITY_PATTERN = re.compile('Q[0-9]+')
    NO_ANSWER = "Not found in our current KB"
    
    neighbor_dict = json.load(open("Datasets/ConvQuestions/neighbors.json"))
    pred_rels = json.load(open("Datasets/ConvQuestions/pred_relations.json"))
    relation_dict = json.load(open("Datasets/ConvQuestions/relations_1hop.json"))
    path_dict = json.load(open("Datasets/ConvQuestions/paths_1hop.json"))
    
    
    eval_path = 'Datasets/ConvQuestions/test_set/q.json'
    all_test_data = json.load(open(eval_path)) 
    total_h1, total_f1, total_cover, total_topic_correct, total = 0.0, 0.0, 0, 0, 0
    
    total_mo,hit1_mo,total_f1_mo = 0,0,0.0
    total_mu,hit1_mu,total_f1_mu = 0,0,0.0
    total_tv,hit1_tv,total_f1_tv = 0,0,0.0
    total_so,hit1_so,total_f1_so = 0,0,0.0
    total_bo,hit1_bo,total_f1_bo = 0,0,0.0
        
    for conv in tqdm(all_test_data):
        s_e_id = re.match(ENTITY_PATTERN, re.sub('^https\:\/\/www\.wikidata\.org\/wiki\/', '', conv["seed_entity"])).group()
        s_e_text = conv["seed_entity_text"]
        domain = conv["domain"]
        entities = {s_e_id:s_e_text}
        
        last_context,last_answer,context = "", "",""
        last_answer_ids,last_answer_texts = [],[]
        for idx,q in enumerate(conv['questions']):
            # # former
            # if idx == 0:
            #     last_context = s_e_text +". "+q['question']+" " if q['question'].endswith("?") else s_e_text +". "+q['question']+"? "
            # else:
            #     last_context += last_answer + q['question']+" " if q['question'].endswith("?") else last_answer + q['question']+"? "
            
            k = s_e_id+" "+q['question']
            context = s_e_text+", "+", ".join(pred_rels[k][0])+". "
            for i in range(0,idx):
                question = conv['questions'][i]['question']
                # pred_answers = last_answer_ids
                # pred_answer_texts = last_answer_texts
                context += question+" " if question.endswith("?") else question+"? "
                for j,a in enumerate(last_answer_ids):
                    k = a+" "+q['question']
                    if re.match(ENTITY_PATTERN, a) and k in pred_rels:
                        context += last_answer_texts[j]+", "+", ".join(pred_rels[k][0])+". "
                    elif re.match(ENTITY_PATTERN, a) and k not in pred_rels:
                        single_pred_rels = get_single_pred_relations(a, q['question'],relation_dict,path_dict,sparql_retriever, relation_retriever)
                        context += last_answer_texts[j]+", "+", ".join(single_pred_rels[k][0])+". "
                    else:
                        context += last_answer_texts[j]+", "
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
            
            te_pred_rels = []
            
            # with rr
            input_q = ""
            candidate_answer = []
            for idx,t in enumerate(te_ids):
                k = t+" "+single_data['raw_question']
                if k in pred_rels:
                    input_q += te_texts[idx]+", "+", ".join(pred_rels[k][0])+". "
                    candidate_answer.append(pred_rels[k][0][1])
                    te_pred_rels.append(pred_rels[k][0][0])
                else:
                    single_pred_rels = get_single_pred_relations(t, single_data['raw_question'],relation_dict,path_dict,sparql_retriever, relation_retriever)
                    input_q += te_texts[idx]+", "+", ".join(single_pred_rels[k][0])+". "
                    candidate_answer.append(single_pred_rels[k][0][1])
                    te_pred_rels.append(single_pred_rels[k][0][0])
                
            input_q += single_data['rewrite_question']
            
            # # without rr
            # input_q = single_data['rewrite_question']
            
            answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
            answers = [re.findall(ENTITY_PATTERN, a)[0] if re.match(ENTITY_PATTERN, a) else a for a in answers]
            answer_texts = [a.strip() for a in q['answer_text'].split(";")]
            if answers[0]=="n/a":answers=answer_texts.copy()
            if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
        
            answers = [formatAnswer(a) for a in answers]
            answers = [id_to_name[re.findall(ENTITY_PATTERN, a)[0]] if re.match(ENTITY_PATTERN, a) else a for a in answers]
            
            assert len(te_texts) == len(te_pred_rels)  
            result, h1, f1 = test_kopl(input_q, answers, te_texts, te_pred_rels)
            
            total_h1 += h1
            total_f1 += f1
            total += 1
            
            if domain=="movies":
                hit1_mo += h1
                total_f1_mo += f1
                total_mo += 1
            elif domain=="tv_series" :
                hit1_tv += h1
                total_f1_tv += f1
                total_tv += 1
            elif domain=="music":
                hit1_mu += h1
                total_f1_mu += f1
                total_mu += 1
            elif domain=="books":
                hit1_bo += h1
                total_f1_bo += f1
                total_bo += 1
            elif domain=="soccer":
                hit1_so += h1
                total_f1_so += f1
                total_so += 1
                    
            if result == NO_ANSWER: 
                last_answer = ", ".join(candidate_answer) +". "
                try:
                    candidate_answer_id = name_to_id[candidate_answer[0]]
                except:
                    candidate_answer_id = sparql_retriever.wikidata_label_to_id(candidate_answer[0])
                    if candidate_answer_id=='UNK':
                        candidate_answer_id = candidate_answer[0]
                last_answer_ids.append(candidate_answer_id)
                last_answer_texts.append(candidate_answer[0])
                #last_answer = s_e_text +". "
                continue
            
            try:
                kopl_answer_id = name_to_id[result]
            except:
                kopl_answer_id = sparql_retriever.wikidata_label_to_id(result)
                if kopl_answer_id == 'UNK':
                    kopl_answer_id = result
            kopl_answer_text = result
            
            # update entities, ner+answer in last turn
            es = [kopl_answer_text] + te_texts
            ids = [kopl_answer_id] + te_ids
            for idx,id in enumerate(ids):
                if id not in entities and re.match(ENTITY_PATTERN, id):
                    entities[id] = es[idx]  
            
            last_answer = kopl_answer_text +". "
            last_answer_ids.append(kopl_answer_id)
            last_answer_texts.append(kopl_answer_text)
            #break
    
    print("h1: ", total_h1/total)
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
    test_final_h1()