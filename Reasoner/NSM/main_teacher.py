import argparse
import sys
from NSM.train.trainer_hybrid import Trainer_KBQA as Trainer_hybrid
from NSM.train.trainer_parallel import Trainer_KBQA as Trainer_parallel
from NSM.util.utils import create_logger
import time
import torch
import numpy as np
import os
import datetime
import copy

import re
import json
from tqdm import tqdm
from preprocessing.parse.con_parse import get_question as con_get_question
from preprocessing.parse.dep_parse import get_question as dep_get_question

from QuestionRewrite.train_rewriter import CQR
from QuestionRewrite.train_relation_retriever import RelationRetriever
from QuestionRewrite.data_processor import processSingleQuestion, initELQ, retrieve_single_subgraph, load_dict, get_single_pred_relations, HiddenPrints
from QuestionRewrite.sparqlretriever import SparqlRetriever



parser = argparse.ArgumentParser()

# datasets
parser.add_argument('--name', default='webqsp', type=str)
parser.add_argument('--model_name', default='rw', type=str)
parser.add_argument('--data_folder', default='/gaolehe/data/KBQA/Freebase/CWQ/', type=str)

# embeddings
parser.add_argument('--word2id', default='vocab_new.txt', type=str)
parser.add_argument('--relation2id', default='relations.txt', type=str)
parser.add_argument('--entity2id', default='entities.txt', type=str)
parser.add_argument('--char2id', default='chars.txt', type=str)
parser.add_argument('--entity_emb_file', default=None, type=str)
parser.add_argument('--entity_kge_file', default=None, type=str)
parser.add_argument('--relation_emb_file', default=None, type=str)
parser.add_argument('--relation_kge_file', default=None, type=str)
parser.add_argument('--word_emb_file', default='word_emb_300d.npy', type=str)
parser.add_argument('--rel_word_ids', default='rel_word_idx.npy', type=str)

# GraftNet embeddings
parser.add_argument('--pretrained_entity_kge_file', default='entity_emb_100d.npy', type=str)

# dimensions, layers, dropout
parser.add_argument('--entity_dim', default=100, type=int)
parser.add_argument('--kge_dim', default=100, type=int)
parser.add_argument('--kg_dim', default=100, type=int)
parser.add_argument('--word_dim', default=300, type=int)
parser.add_argument('--lstm_dropout', default=0.3, type=float)
parser.add_argument('--linear_dropout', default=0.2, type=float)

# optimization
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--fact_scale', default=3, type=int)
parser.add_argument('--eval_every', default=5, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--gradient_clip', default=1.0, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--decay_rate', default=0.0, type=float)
parser.add_argument('--seed', default=19960626, type=int)
parser.add_argument('--lr_schedule', action='store_true')
parser.add_argument('--label_smooth', default=0.1, type=float)
parser.add_argument('--fact_drop', default=0, type=float)

# model options
parser.add_argument('--q_type', default='seq', type=str)
parser.add_argument('--share_encoder', action='store_true')
parser.add_argument('--use_inverse_relation', action='store_true')
parser.add_argument('--use_self_loop', action='store_true')
parser.add_argument('--train_KL', action='store_true')
parser.add_argument('--is_eval', action='store_true')
parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str)
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--experiment_name', default='debug', type=str)
parser.add_argument('--load_experiment', default=None, type=str)
parser.add_argument('--load_pretrain', default=None, type=str)
parser.add_argument('--eps', default=0.05, type=float) # threshold for f1

# RL options
parser.add_argument('--filter_sub', action='store_true')
parser.add_argument('--encode_type', action='store_true')
parser.add_argument('--reason_kb', action='store_true')
parser.add_argument('--num_layer', default=1, type=int)
parser.add_argument('--test_batch_size', default=20, type=int)
parser.add_argument('--num_step', default=1, type=int)
parser.add_argument('--mode', default='teacher', type=str)
parser.add_argument('--entropy_weight', default=0.0, type=float)

parser.add_argument('--use_label', action='store_true')
parser.add_argument('--tree_soft', action='store_true')
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--share_instruction', action='store_true')
parser.add_argument('--filter_label', action='store_true')
parser.add_argument('--teacher_type', default='hybrid', type=str)
parser.add_argument('--encoder_type', default='lstm', type=str)
parser.add_argument('--constrain_type', default='mse', type=str)
parser.add_argument('--lambda_label', default=0.01, type=float)
parser.add_argument('--lambda_constrain', default=0.1, type=float)
parser.add_argument('--lambda_back', default=0.01, type=float)
parser.add_argument('--label_f1', default=0.5, type=float)
parser.add_argument('--loss_type', default='kl', type=str)
parser.add_argument('--label_file', default=None, type=str)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )

'''
def main():
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    if args.teacher_type == "parallel":
        trainer = Trainer_parallel(args=vars(args), logger=logger)
    elif args.teacher_type == "hybrid":
        trainer = Trainer_hybrid(args=vars(args), logger=logger)
    else:
        raise NotImplementedError
    if not args.is_eval:
        trainer.train(0, args.num_epoch - 1)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)
'''
def main():
    if not args.is_eval:
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        logger = create_logger(args)
        if args.teacher_type == "parallel":
            trainer = Trainer_parallel(args=vars(args), logger=logger)
        elif args.teacher_type == "hybrid":
            trainer = Trainer_hybrid(args=vars(args), logger=logger)
        else:
            raise NotImplementedError
        trainer.train(0, args.num_epoch - 1)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        
        config_fn = 'QuestionRewrite/config_ConvQuestions_selftrain.json'
        print("Loading finetuned t5 model from {}".format(config_fn))
        with open(config_fn) as f:
            t5_args = json.load(f)
        cqr = CQR(t5_args)  
        
        config_fn = 'QuestionRewrite/config_relation_retriever.json'
        print("Loading relation retriever model from {}".format(config_fn))
        with open(config_fn) as f:
            rr_args = json.load(f)
        relation_retriever = RelationRetriever(rr_args)
        
        elq_args, models, id2wikidata = initELQ()
        sparql_dir = '../../KB-cache'
        sparql_retriever = SparqlRetriever()
        sparql_retriever.load_cache('%s/M2N.json' % sparql_dir,
                                '%s/STATEMENTS.json' % sparql_dir,
                                '%s/QUERY.json' % sparql_dir,
                                '%s/TYPE.json' % sparql_dir,
                                '%s/OUTDEGREE.json' % sparql_dir)
        
        entity2id = load_dict("ConvQuestions/entities.txt")
        id2entity = {idx: entity for entity, idx in entity2id.items()}
        relation2id = load_dict("ConvQuestions/relations.txt") 
        relation_dict = json.load(open("../../Datasets/ConvQuestions/relations_1hop.json"))
        path_dict = json.load(open("../../Datasets/ConvQuestions/paths_1hop.json"))
        neighbor_dict = json.load(open("../../Datasets/ConvQuestions/neighbors.json"))
        pred_rels = json.load(open("../../Datasets/ConvQuestions/pred_relations.json"))
        
        ENTITY_PATTERN = re.compile('Q[0-9]+')
        
        eval_path = 'QuestionRewrite/q.json'
        file_path = 'ConvQuestions/test_simple.json'
        all_test_data = json.load(open(eval_path)) 
        total_h1, total_f1, total_cover, total_cover_, total_topic_correct, total = 0.0, 0.0, 0, 0, 0, 0
        
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
            
            last_context,last_answer = "", ""
            last_answer_ids,last_answer_texts = [],[]
            for idx,q in enumerate(conv['questions']):
                # if idx == 0:
                #     last_context = s_e_text +". "+q['question']+" " if q['question'].endswith("?") else s_e_text +". "+q['question']+"? "
                # else:
                #     last_context += last_answer + q['question']+" " if q['question'].endswith("?") else last_answer + q['question']+"? "
                
                k = s_e_id+" "+q['question']
                context = s_e_text+", "+", ".join(pred_rels[k][0])+". "
                for i in range(0,idx):
                    question = conv['questions'][i]['question']
                    answers = last_answer_ids
                    answer_texts = last_answer_texts
                    context += question+" " if question.endswith("?") else question+"? "
                    for j,a in enumerate(answers):
                        k = a+" "+q['question']
                        if re.match(ENTITY_PATTERN, a) and k in pred_rels:
                            context += answer_texts[j]+", "+", ".join(pred_rels[k][0])+". "
                        elif re.match(ENTITY_PATTERN, a) and k not in pred_rels:
                            single_pred_rels = get_single_pred_relations(a, q['question'],relation_dict,path_dict,sparql_retriever, relation_retriever)
                            context += answer_texts[j]+", "+", ".join(single_pred_rels[k][0])+". "
                        else:
                            context += answer_texts[j]+", "
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
                    check_question,te_ids,te_texts = processSingleQuestion(elq_args, models, id2wikidata, single_data, sparql_retriever, neighbor_dict, entity2id)
                single_data['rewrite_question'] = check_question
                single_data['te_ids']           = te_ids
                single_data['te_texts']         = te_texts
                
                cover, cover_ = retrieve_single_subgraph(single_data, entity2id, relation2id, path_dict, sparql_retriever, file_path)
                total_cover += cover
                total_cover_ += cover_
                
                con_get_question(file_path, 'ConvQuestions/test.con')
                dep_get_question(file_path, 'ConvQuestions/test.dep')
                
                with HiddenPrints():
                    logger = create_logger(args)
                    if args.teacher_type == "parallel":
                        trainer = Trainer_parallel(args=vars(args), logger=logger)
                    elif args.teacher_type == "hybrid":
                        trainer = Trainer_hybrid(args=vars(args), logger=logger)
                    else:
                        raise NotImplementedError
                    f1, h1, nsm_answer_id = trainer.evaluate_single(ckpt_path)
                
                total_f1 += f1
                total_h1 += h1
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
                
                
                nsm_answer_id = id2entity[nsm_answer_id]
                if re.match(ENTITY_PATTERN, nsm_answer_id):
                    nsm_answer_text = sparql_retriever.wikidata_id_to_label(nsm_answer_id)
                else:
                    nsm_answer_text = nsm_answer_id
                
                # update entities, ner+answer in last turn
                es = [nsm_answer_text] + te_texts
                ids = [nsm_answer_id] + te_ids
                for idx,id in enumerate(ids):
                    if id not in entities and re.match(ENTITY_PATTERN, id):
                        entities[id] = es[idx]  
                
                last_answer = nsm_answer_text +". "
                
                last_answer_ids.append(nsm_answer_id)
                last_answer_texts.append(nsm_answer_text)
                #break
                
        print("h1: ", total_h1/total)
        print("f1: ", total_f1/total)
        print("cover: ", total_cover/total)
        print("cover_: ", total_cover_/total)
        # print("topic: ", total_topic_correct/total)
        
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
    main()
