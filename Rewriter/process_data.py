
import os
import re
import json
from tqdm import tqdm
from utils import load_sparql_retriever

# question types
TOTAL               = 'total'
OVERALL             = 'Overall'
CLARIFICATION       = 'Clarification'
COMPARATIVE         = 'Comparative Reasoning (All)'
LOGICAL             = 'Logical Reasoning (All)'
QUANTITATIVE        = 'Quantitative Reasoning (All)'
SIMPLE_COREFERENCED = 'Simple Question (Coreferenced)'
SIMPLE_DIRECT       = 'Simple Question (Direct)'
SIMPLE_ELLIPSIS     = 'Simple Question (Ellipsis)'
VERIFICATION        = 'Verification (Boolean) (All)'
QUANTITATIVE_COUNT  = 'Quantitative Reasoning (Count) (All)'
COMPARATIVE_COUNT   = 'Comparative Reasoning (Count) (All)'


def process_CANARD(fpath):
    cw_pairs = []
    with open(fpath, "r") as f:
        all_data = json.load(f)
        for cq in all_data:
            context = ""
            utterances = cq['History']
            for idx,u in enumerate(utterances):
                if idx==0:
                    context += u+" " if u.endswith(".") else u+". "
                    continue
                #if idx==1:continue #ignore the topic description
                if idx%2==0:#question
                    context += u+" " if u.endswith("?") else u+"? "
                else:
                    context += u+" " if (u.endswith(".") or u.endswith(",") or u.endswith(":") or u.endswith(";")) else u+". "
            question = cq['Question']
            context += question if question.endswith("?") else question+"?"
            rewrite = cq['Rewrite'] if cq['Rewrite'].endswith("?") else cq['Rewrite']+"?"
            #print(context, rewrite)
            cw_pairs.append([context, rewrite])
            
    return cw_pairs


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




if __name__ == '__main__':
    
    data_dir = "../Datasets/CANARD"
    train_path = os.path.join(data_dir, 'train.json')
    process_CANARD(train_path)
    
    
    
    