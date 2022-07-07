
import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import random
import json
from tqdm import tqdm
from datetime import date
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import torch.optim as optim
import logging
import time
import re
import logging
import warnings
sys.path.append('/home/xxx/CQR/Reasoner/KoPL/code')
from kopl.kopl import KoPLEngine
sys.path.append('/home/xxx/CQR/Rewriter')
from sparqlretriever import SparqlRetriever


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain",            action='store_true')
parser.add_argument("--selftrain",           action='store_true')
args = parser.parse_args()
if args.pretrain:
    model_path = '/home/xxx/CQR/Reasoner/KoPL/Question2KoPL'
elif args.selftrain:
    model_path = '/home/xxx/CQR/Reasoner/KoPL/code/kopl_selftrain_rr'
else:
    model_path = '/home/xxx/CQR/Reasoner/KoPL/code/kopl_selftrain_rr'
    print("default model path: kopl_selftrain_rr")


kb_path = '/home/xxx/CQR/Reasoner/KoPL/KB/wikidata.json'
rel_path = "../../../Datasets/ConvQuestions/pred_relations.json"


def load_model(model_path, kb_path):

    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    program_tokenizer = tokenizer_class.from_pretrained(model_path)
    program_model = model_class.from_pretrained(model_path)

    program_model = program_model.to(device)
    print('loading kb')
    kb = json.load(open(kb_path))
    print('kb loaded')
    program_rule_executor = KoPLEngine(kb)

    return program_tokenizer, program_model, program_rule_executor, device


program_tokenizer, program_model, program_rule_executor, device = load_model(model_path,kb_path)


def program_vis(text):
    pattern = re.compile(r'(.*?)\((.*?)\)')
    def get_dep(program, inputs):
        program = ['<START>'] + program + ['<END>']
        inputs = [[]] + inputs + [[]]
        dependency = []
        branch_stack = []
        for i, p in enumerate(program):
            if p in {'<START>', '<END>', '<PAD>'}:
                dep = [0, 0]
            elif p in {'FindAll', 'Find'}:
                dep = [0, 0]
                branch_stack.append(i - 1)
            elif p in {'And', 'Or', 'SelectBetween', 'QueryRelation', 'QueryRelationQualifier'}:
                dep = [branch_stack[-1], i-1]
                branch_stack = branch_stack[:-1]
            else:
                dep = [i-1, 0]
            dependency.append(dep)

        assert len(program) == len(inputs)
        assert len(program) == len(dependency)
        for i in range(len(dependency)):
            dependency[i] = [dependency[i][0] - 1, dependency[i][1] - 1]
        return dependency[1:-1]

    with torch.no_grad():
        input_ids = program_tokenizer.batch_encode_plus([text], max_length = 512, pad_to_max_length = True, return_tensors="pt", truncation = True)
        source_ids = input_ids['input_ids'].to(device)
        outputs = program_model.generate(
            input_ids=source_ids,
            max_length = 500
        )
        outputs = [program_tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs]
        output = outputs[0]
        
        filter_pattern = re.compile("FilterConcept(.*?)<b>")
        output = filter_pattern.sub("",output,)
        
        chunks = output.split('<b>')
        func_list = []
        inputs_list = []
        for chunk in chunks:
            res = pattern.findall(chunk)
            if len(res) == 0:
                continue
            res = res[0]
            func, inputs = res[0], res[1]
            if inputs == '':
                inputs = []
            else:
                inputs = inputs.split('<c>')
            func_list.append(func)
            inputs_list.append(inputs)
        assert len(func_list) == len(inputs_list)
        
        dep_list = get_dep(func_list, inputs_list)
        assert len(dep_list) == len(func_list)
        assert len(func_list) == len(inputs_list)
    program = []
    for func, inputs, dep in zip(func_list, inputs_list, dep_list):
        if func=="FilterConcept":continue
        program.append({'func': func, 'inputs': inputs, 'dep': dep})
    return program, func_list, inputs_list, output


def program_vis_batch(text):
    pattern = re.compile(r'(.*?)\((.*?)\)')
    def get_dep(program, inputs):
        program = ['<START>'] + program + ['<END>']
        inputs = [[]] + inputs + [[]]
        dependency = []
        branch_stack = []
        for i, p in enumerate(program):
            if p in {'<START>', '<END>', '<PAD>'}:
                dep = [0, 0]
            elif p in {'FindAll', 'Find'}:
                dep = [0, 0]
                branch_stack.append(i - 1)
            elif p in {'And', 'Or', 'SelectBetween', 'QueryRelation', 'QueryRelationQualifier'}:
                dep = [branch_stack[-1], i-1]
                branch_stack = branch_stack[:-1]
            else:
                dep = [i-1, 0]
            dependency.append(dep)
        assert len(program) == len(inputs)
        assert len(program) == len(dependency)
        for i in range(len(dependency)):
            dependency[i] = [dependency[i][0] - 1, dependency[i][1] - 1]
        return dependency[1:-1]

    with torch.no_grad():
        input_ids = program_tokenizer.batch_encode_plus(text, max_length = 512, pad_to_max_length = True, return_tensors="pt", truncation = True)
        source_ids = input_ids['input_ids'].to(device)
        source_attn_mask = input_ids['attention_mask'].to(device)
        outputs = program_model.generate(
            input_ids=source_ids,
            attention_mask=source_attn_mask,
            max_length = 500,
        )
        outputs = program_tokenizer.batch_decode(outputs, skip_special_tokens = True, clean_up_tokenization_spaces = True)
    
    for output in outputs:    
        filter_pattern = re.compile("FilterConcept(.*?)<b>")
        output = filter_pattern.sub("",output,)
        
        chunks = output.split('<b>')
        func_list = []
        inputs_list = []
        for chunk in chunks:
            res = pattern.findall(chunk)
            if len(res) == 0:
                continue
            res = res[0]
            func, inputs = res[0], res[1]
            if inputs == '':
                inputs = []
            else:
                inputs = inputs.split('<c>')
            func_list.append(func)
            inputs_list.append(inputs)
        assert len(func_list) == len(inputs_list)
        dep_list = get_dep(func_list, inputs_list)
        assert len(dep_list) == len(func_list)
        assert len(func_list) == len(inputs_list)
        
        program = []
        for func, inputs, dep in zip(func_list, inputs_list, dep_list):
            if func=="FilterConcept":continue
            program.append({'func': func, 'inputs': inputs, 'dep': dep})
        yield program, func_list, inputs_list, output


def infer_kpol_batch(question):
    program_results, kopl_queries = [], []
    for program, func_list, inputs_list, kopl_query in program_vis_batch(question):
        program_result = program_rule_executor.forward(func_list, inputs_list, ignore_error = True, show_details = False)
        if isinstance(program_result, list) and len(program_result) > 0:
            program_result = program_result[0]
        if program_result is None or program_result == 'None' or program_result == []:
            program_result = 'Not found in our current KB'
        
        program_results.append(program_result)
        kopl_queries.append(kopl_query)
        
    return program_results, kopl_queries


def infer_convex_batch(data_dir):
    for dataset in ['train_set', 'dev_set', 'test_set']:
        with open(os.path.join(data_dir, dataset, 'rewrite_q_selftrain_rr_elq.json')) as qwFile:
            questions = []
            all_data = json.load(qwFile)
            for conv in tqdm(all_data):
                for q in conv['questions']:
                    question = q['rewrite']
                    questions.append(question)
        
            batch_size = 512
            results, kopl_queries = [], []
            for idx in tqdm(range(0,len(questions),batch_size)):
                batch_q = questions[idx:idx+batch_size]
                result, kopl_query = infer_kpol_batch(batch_q)
                results += result
                kopl_queries += kopl_query
        
            for i,conv in enumerate(tqdm(all_data)):
                for j,q in enumerate(conv['questions']):    
                    q['kpol_ans'] = results[5*i+j]
                    q['kpol_query'] = kopl_queries[5*i+j]
                    
        json.dump(all_data, open(dataset+'_kopl_selftrain_rr.json', "w"), indent=4)



def generate_selftrain_data():
    pred_rels = json.load(open(rel_path))
    
    id_to_name = json.load(open("../KB/id_to_name.json"))
    ENTITY_PATTERN = re.compile('Q[0-9]+')
    NO_ANSWER = "Not found in our current KB"
    
    h1_, total_ = 0, 0
    for dataset in ['train_set', 'dev_set', 'test_set']:
        h1, total = 0, 0
        qk_pairs = []
        with open(dataset+'_kopl_selftrain_rr.json') as qwFile:
            all_data = json.load(qwFile)
            for conv in tqdm(all_data):
                for q in conv['questions']:
                    answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
                    answer_texts = [a.strip() for a in q['answer_text'].split(";")]
                    if answers[0]=="n/a":answers=answer_texts.copy()
                    if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
                
                    answers = [formatAnswer(a) for a in answers]
                    answers = [id_to_name[re.findall(ENTITY_PATTERN, a)[0]] if re.match(ENTITY_PATTERN, a) else a for a in answers]
                
                    kpol_result = [formatAnswer(q['kpol_ans'])]
                    
                    if is_date(kpol_result[0]):kpol_result.append(kpol_result[0].split(" ")[2])
                    
                    if isExistential(q['question'].split(" ")[0].lower()):
                        if kpol_result[0]!=NO_ANSWER:
                            kpol_result = ['yes']
                        else:
                            kpol_result = ['no']
                    
                    te_ids = q['te_ids'].split(";")
                    te_texts = q['te_texts'].split(";")
                    for r in kpol_result:
                        if r in answers:
                            for idx,t in enumerate(te_ids):
                                k = t+" "+q['question']
                                pred_rel = pred_rels[k][0][0]
                                if te_texts[idx].lower() in q['rewrite'].lower() and te_texts[idx].lower() in q['kpol_query'].lower() and pred_rel in q['kpol_query']:
                                    qk_pairs.append([q['rewrite'], q['kpol_query'], q['question'], q['te_ids'], q['te_texts']])
                                    h1 += 1
                                    h1_ += 1
                                    break
                            break
                    total += 1
                    total_ += 1
        print(h1, total, h1/total)
        with open(dataset+'_qk_st_rr.txt','w') as f:
            for qk in qk_pairs:
                f.write(str(qk)+"\n")
                
    print(h1_, total_, h1_/total_)



def generate_selftrain_data_union():
    pred_rels = json.load(open(rel_path))
    
    id_to_name = json.load(open("../KB/id_to_name.json"))
    ENTITY_PATTERN = re.compile('Q[0-9]+')
    NO_ANSWER = "Not found in our current KB"
    
    h1_, total_ = 0, 0
    for dataset in ['train_set', 'dev_set', 'test_set']:
        h1, total = 0, 0
        pt_qk_pairs = []
        st_qk_pairs = []
        rr_qk_pairs = []
        
        pt_data = json.load(open(dataset+'_kopl_pretrain.json'))
        st_data = json.load(open(dataset+'_kopl_selftrain.json'))
        rr_data = json.load(open(dataset+'_kopl_selftrain_rr.json'))
        for c_idx, conv in enumerate(pt_data):
            for q_idx, q in enumerate(conv['questions']):
                answers = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in re.sub('\.$', '', q['answer']).split(';')]
                answer_texts = [a.strip() for a in q['answer_text'].split(";")]
                if answers[0]=="n/a":answers=answer_texts.copy()
                if len(answers)!=len(answer_texts):answer_texts=answer_texts[:len(answers)]
            
                answers = [formatAnswer(a) for a in answers]
                answers = [id_to_name[re.findall(ENTITY_PATTERN, a)[0]] if re.match(ENTITY_PATTERN, a) else a for a in answers]
                
                pt_rewrite = q['rewrite']
                st_rewrite = st_data[c_idx]['questions'][q_idx]['rewrite']
                rr_rewrite = rr_data[c_idx]['questions'][q_idx]['rewrite']
                rewrites = [pt_rewrite, st_rewrite, rr_rewrite]
                
                pt_kopl_query = q['kpol_query']
                st_kopl_query = st_data[c_idx]['questions'][q_idx]['kpol_query']
                rr_kopl_query = rr_data[c_idx]['questions'][q_idx]['kpol_query']
                queries = [pt_kopl_query, st_kopl_query, rr_kopl_query]
                
                pt_kopl_ans = q['kpol_ans']
                st_kopl_ans = st_data[c_idx]['questions'][q_idx]['kpol_ans']
                rr_kopl_ans = rr_data[c_idx]['questions'][q_idx]['kpol_ans']
                
                pt_te_ids   = q['te_ids'].split(";")
                pt_te_texts = q['te_texts'].split(";")
                st_te_ids   = st_data[c_idx]['questions'][q_idx]['te_ids'].split(";")
                st_te_texts = st_data[c_idx]['questions'][q_idx]['te_texts'].split(";")
                rr_te_ids   = rr_data[c_idx]['questions'][q_idx]['te_ids'].split(";")
                rr_te_texts = rr_data[c_idx]['questions'][q_idx]['te_texts'].split(";")
                tids = [pt_te_ids, st_te_ids, rr_te_ids]
                ttexts = [pt_te_texts, st_te_texts, rr_te_texts]
                
                flag = 0
                for i, kpol_result in enumerate([pt_kopl_ans, st_kopl_ans, rr_kopl_ans]):
                
                    kpol_result = [formatAnswer(kpol_result)]
                    
                    if is_date(kpol_result[0]):kpol_result.append(kpol_result[0].split(" ")[2])
                    
                    if isExistential(q['question'].split(" ")[0].lower()):
                        if kpol_result[0]!=NO_ANSWER:
                            kpol_result = ['yes']
                        else:
                            kpol_result = ['no']
                    
                    te_ids = tids[i]
                    te_texts = ttexts[i]
                    for r in kpol_result:
                        if r in answers:
                            for idx,t in enumerate(te_ids):
                                k = t+" "+q['question']
                                pred_rel = pred_rels[k][0][0]
                                if te_texts[idx].lower() in rewrites[i].lower() and te_texts[idx].lower() in queries[i].lower() and pred_rel in queries[i]:
                                    pt_qk_pairs.append([pt_rewrite, pt_kopl_query, q['question'], ";".join(pt_te_ids), ";".join(pt_te_texts)])
                                    st_qk_pairs.append([st_rewrite, st_kopl_query, q['question'], ";".join(st_te_ids), ";".join(st_te_texts)])
                                    rr_qk_pairs.append([rr_rewrite, rr_kopl_query, q['question'], ";".join(rr_te_ids), ";".join(rr_te_texts)])
                                    flag = 1
                                    break
                                
                            h1 += 1
                            h1_ += 1
                            break
                    
                    if flag==1:break
                    
                total += 1
                total_ += 1
        print(h1, total, h1/total)
        with open(dataset+'_qk_pt.txt','w') as f:
            for qk in pt_qk_pairs:
                f.write(str(qk)+"\n")
        
        with open(dataset+'_qk_st.txt','w') as f:
            for qk in st_qk_pairs:
                f.write(str(qk)+"\n")
        
        with open(dataset+'_qk_st_rr.txt','w') as f:
            for qk in rr_qk_pairs:
                f.write(str(qk)+"\n")
                
    print(h1_, total_, h1_/total_)


def rr_modify_kopl(func_list, inputs_list, te_texts, te_pred_rels):
    if "Find" not in func_list:
        for idx, t in enumerate(te_texts):
            f_l = ["Find", "Relate", "What"]
            i_l = [[t], [te_pred_rels[idx], 'forward'],[]]
            program_result = program_rule_executor.forward(f_l, i_l, ignore_error = True, show_details = False)
            if isinstance(program_result, list) and len(program_result) > 0:
                return program_result[0]
            
            f_l = ["Find", "QueryAttr"]
            i_l = [[t], [te_pred_rels[idx]]]
            program_result = program_rule_executor.forward(f_l, i_l, ignore_error = True, show_details = False)
            if isinstance(program_result, list) and len(program_result) > 0:
                return program_result[0]
            
        return "Not found in our current KB"
    
    k_t, k_r = "", ""
    for idx, t in enumerate(te_texts):
        if t in inputs_list[func_list.index("Find")] and len(t)>len(k_t):
            k_t = t
            k_r = te_pred_rels[idx]

    f = ""
    if "Relate" in func_list:
        f = "Relate"
    elif "QueryAttr" in func_list:
        f = "QueryAttr"
    else:
        return "Not found in our current KB"
    
    inputs_list[func_list.index(f)][0] = k_r
    program_result = program_rule_executor.forward(func_list, inputs_list, ignore_error = True, show_details = False)
    if isinstance(program_result, list) and len(program_result) > 0:
        return program_result[0]
    
    f_l, i_l = [], []
    if "Relate" in func_list:
        f_l = ["Find", "Relate", "What"]
        i_l = [[k_t], [k_r, 'forward'],[]]
    elif "QueryAttr" in func_list:
        f_l = ["Find", "QueryAttr"]
        i_l = [[k_t], [k_r]]
    program_result = program_rule_executor.forward(f_l, i_l, ignore_error = True, show_details = False)
    if isinstance(program_result, list) and len(program_result) > 0:
        return program_result[0]
    
    return "Not found in our current KB"
     

def test_kopl(question, answers, te_texts, te_pred_rels):
    NO_ANSWER = "Not found in our current KB"
    
    program, func_list, inputs_list, kpol_query = program_vis(question)
    program_result = program_rule_executor.forward(func_list, inputs_list, ignore_error = True, show_details = False)
    if isinstance(program_result, list) and len(program_result) > 0:
        program_result = program_result[0]
    if program_result is None or program_result == 'None' or program_result == []:
        # program_result = NO_ANSWER
        program_result = rr_modify_kopl(func_list, inputs_list, te_texts, te_pred_rels)
    
    kpol_result = [formatAnswer(program_result)]
    
    if isExistential(question.split(" ")[0].lower()):
        if kpol_result[0]!=NO_ANSWER:
            kpol_result = ['yes','no']
        else:
            kpol_result = ['no','yes']
    else:
        for kr in kpol_result:
            if is_date(kr):
                kpol_result.append(kr.split(" ")[2])
    
    h1 = 0
    if kpol_result[0] in answers:
        h1 = 1
        
    correct = 0
    for r in kpol_result:
        if r in answers:
            correct += 1
    
    p, r = correct / len(kpol_result), correct / len(answers)
    f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
    
    return kpol_result[0], h1, f1
        

def convertMonth(month):
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

def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]')
    if not(pattern.match(timestamp)):
        return False
    else:
        return True

def is_date(date):
    pattern = re.compile('^[0-9]+ [A-z]+ [0-9][0-9][0-9][0-9]$')
    if not(pattern.match(date.strip())):
        return False
    else:
        return True

def convertTimestamp( timestamp):
    yearPattern = re.compile('^[0-9][0-9][0-9][0-9]-00-00')
    monthPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-00')
    dayPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]')
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

def formatAnswer(answer):
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

def isExistential(question_start):
    existential_keywords = ['is', 'are', 'was', 'were', 'am', 'be', 'being', 'been', 'did', 'do', 'does', 'done', 'doing', 'has', 'have', 'had', 'having']
    if question_start in existential_keywords:
        return True
    return False
    

def get_answer_text(data_dir, sparql_retriever):
    ENTITY_PATTERN = re.compile('Q[0-9]+')
    items = json.load(open("../KB/item.json"))

    id_to_name = {}
    for dataset in ['train_set', 'dev_set', 'test_set']:
        fpath = os.path.join(data_dir, dataset, 'q.json')
        all_data = json.load(open(fpath))
        for conv in all_data:
            for q in conv['questions']:
                answer_id = [re.sub('^[ ]*https\:\/\/www\.wikidata\.org\/wiki\/', '', a) for a in q['answer'].split(";")]
                for a in answer_id:
                    if re.match(ENTITY_PATTERN, a):
                        #print(a)
                        a = re.findall(ENTITY_PATTERN, a)[0]
                        #print(a)
                        if a not in id_to_name:
                            id_to_name[a] = sparql_retriever.wikidata_id_to_label(a)
                            if id_to_name[a] == "[UNK]":
                                try:
                                    id_to_name[a] = items[a]
                                except:
                                    pass
    json.dump(id_to_name, open("../KB/id_to_name.json", "w"))   



if __name__ == '__main__':
    if args.pretrain:
        data_dir = "../../../Datasets/ConvQuestions"
        sparql_dir = '../../../KB-cache/'
        sparql_retriever = SparqlRetriever()
        sparql_retriever.load_cache('%s/M2N.json' % sparql_dir,
                                '%s/STATEMENTS.json' % sparql_dir,
                                '%s/QUERY.json' % sparql_dir,
                                '%s/TYPE.json' % sparql_dir,
                                '%s/OUTDEGREE.json' % sparql_dir)

        infer_convex_batch(data_dir)
        get_answer_text(data_dir, sparql_retriever)
        generate_selftrain_data()
    
    
    
