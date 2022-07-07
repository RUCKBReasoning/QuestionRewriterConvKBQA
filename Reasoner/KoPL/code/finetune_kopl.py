
import os
import re
import ast
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import set_seed
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def process_CONVEX_rel(fpath, pred_rels):
    qk_pairs = []
    with open(fpath, "r") as f:
        for line in f :
            input_q = ""
            rewrite_q, kopl_q, raw_q, te_ids, te_texts  = ast.literal_eval(line.strip())
            
            te_ids = te_ids.split(";")
            te_texts = te_texts.split(";")
                
            for idx,t in enumerate(te_ids):
                k = t+" "+raw_q
                input_q += te_texts[idx]+", "+", ".join(pred_rels[k][0])+". "
                    
            input_q += rewrite_q
            qk_pairs.append([input_q,kopl_q])      
    return qk_pairs


def process_CONVEX(fpath):
    qk_pairs = []
    with open(fpath, "r") as f:
        for line in f :
            rewrite_q, kopl_q, raw_q, te_ids, te_texts  = ast.literal_eval(line.strip())
            qk_pairs.append([rewrite_q,kopl_q])      
    return qk_pairs


class CQRDataset(Dataset):
    def __init__(self, encodings, bad_indexes):
        self.encodings   = encodings
        self.bad_indexes = bad_indexes

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.encodings.items()}


class T2TDataCollator:
    def __call__(self, batch):
        input_ids = torch.stack([example['input_ids']  for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 1] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': lm_labels, 'decoder_attention_mask': decoder_attention_mask }
        

class CQR():
    def __init__(self, args) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        args['train_args']['local_rank'] = self.local_rank

        # General arguments
        self.gen_args          = args['gen_args']
        self.model_path        = self.gen_args['model_path']
        self.dataset_dir       = self.gen_args['dataset_dir']
        self.result_fn         = self.gen_args['result_fn']
        self.train_fn          = self.gen_args['train_fn']
        self.valid_fn          = self.gen_args['valid_fn']
        self.test_fn           = self.gen_args['test_fn']
        self.rel_fn            = self.gen_args['pred_rels_fn']
        self.max_in_len        = self.gen_args['max_in_len']
        self.max_out_len       = self.gen_args['max_out_len']
        self.max_candidates    = self.gen_args['max_candidates']
        
        # HuggingFace trainer arguments
        self.training_args = TrainingArguments(**args['train_args'])
        set_seed(self.training_args.seed)
        
        # T5 model
        self.tokenizer = BartTokenizer.from_pretrained(self.model_path)
        
        # parallel
        # if args['train_args']['do_eval']:
        #     self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        # else:
        #     self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)

        # single gpu
        self.model = BartForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        #self.model.resize_token_embeddings(len(self.tokenizer))
        
        
        # print('Building datasets')
        # train_file_path = os.path.join(self.dataset_dir, self.train_fn)
        # valid_file_path = os.path.join(self.dataset_dir, self.valid_fn)
        # self.train_dataset = self.build_dataset(train_file_path)
        # print('Training data is {:,} after removing {:,} long entries'.format(len(self.train_dataset), len(self.train_dataset.bad_indexes)))
        # self.valid_dataset = self.build_dataset(valid_file_path)
        # print('Validation data is {:,} after removing {:,} long entries'.format(len(self.valid_dataset), len(self.valid_dataset.bad_indexes)))
        
        
        # no_decay = ["bias", "LayerNorm.weight"]
        # bart_param_optimizer = list(self.model.named_parameters())
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
        #         'weight_decay': self.training_args.weight_decay, 'lr': self.training_args.learning_rate},
        #     {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        #         'lr': self.training_args.learning_rate}
        # ]
        # t_total = len(self.train_dataset)
        # warmup_steps = int(t_total * self.training_args.warmup_ratio)
        # self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate, eps=self.training_args.adam_epsilon)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,num_training_steps=t_total)
        # if os.path.isfile(os.path.join(self.model_path, "optimizer.pt")) and os.path.isfile(
        #         os.path.join(self.model_path, "scheduler.pt")):
        #     # Load in optimizer and scheduler states
        #     self.optimizer.load_state_dict(torch.load(os.path.join(self.model_path, "optimizer.pt")))
        #     self.scheduler.load_state_dict(torch.load(os.path.join(self.model_path, "scheduler.pt")))
        

    def build_dataset(self, fpath):
        # Load the raw data
        # qa_pairs = process_CONVEX(fpath)
        pred_rels = json.load(open(self.rel_fn))
        qa_pairs = process_CONVEX_rel(fpath, pred_rels)
        random.shuffle(qa_pairs)

        # Convert to input and target sentences
        input_text  = ['%s' % qa[0] for qa in qa_pairs]
        target_text = ['%s' % qa[1] for qa in qa_pairs]
        
        # Form the input encodings
        print('Batch encoding')
        input_encodings  = self.tokenizer.batch_encode_plus(input_text,
                            padding=True, truncation=True, max_length=self.max_in_len,
                            return_overflowing_tokens=True)
        target_encodings = self.tokenizer.batch_encode_plus(target_text,
                            padding=True, truncation=True, max_length=self.max_out_len,
                            return_overflowing_tokens=True)
        
        # Remove any sens that are greater than max length after tokenization
        # Find the bad indexes
        bi = set()
        for i, (ie, te) in enumerate(zip(input_encodings['num_truncated_tokens'], target_encodings['num_truncated_tokens'])):
            if ie > 0 or te > 0:
                bi.add( i )
        
        # Remove them
        input_encodings['input_ids']       = [ie for i, ie in enumerate(input_encodings['input_ids'])       if i not in bi]
        target_encodings['input_ids']      = [te for i, te in enumerate(target_encodings['input_ids'])      if i not in bi]
        input_encodings['attention_mask']  = [ie for i, ie in enumerate(input_encodings['attention_mask'])  if i not in bi]
        target_encodings['attention_mask'] = [te for i, te in enumerate(target_encodings['attention_mask']) if i not in bi]
        
        # Create the encodings
        encodings = {'input_ids':             torch.LongTensor(input_encodings['input_ids']),
                     'attention_mask':        torch.LongTensor(input_encodings['attention_mask']),
                     'target_ids':            torch.LongTensor(target_encodings['input_ids']),
                     'target_attention_mask': torch.LongTensor(target_encodings['attention_mask']) }
        # Encapsulate the data and return
        return CQRDataset(encodings, bi)
    

    def train(self):
        
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        
        print('Building datasets')
        train_file_path = os.path.join(self.dataset_dir, self.train_fn)
        valid_file_path = os.path.join(self.dataset_dir, self.valid_fn)
        train_dataset = self.build_dataset(train_file_path)
        print('Training data is {:,} after removing {:,} long entries'.format(len(train_dataset), len(train_dataset.bad_indexes)))
        valid_dataset = self.build_dataset(valid_file_path)
        print('Validation data is {:,} after removing {:,} long entries'.format(len(valid_dataset), len(valid_dataset.bad_indexes)))
        
        # Train the model
        print('Training')
        trainer = Trainer(model=self.model,
                          args=self.training_args,
                          train_dataset=train_dataset,
                          eval_dataset=valid_dataset,
                          data_collator=T2TDataCollator())
        trainer.train()
        
        # Save the results
        print('Saving model')
        trainer.save_model(self.training_args.output_dir)

    
    def next_batch_data(self, all_data):
        for idx in tqdm(range(0,len(all_data),self.training_args.per_device_eval_batch_size)):
            batch_data = all_data[idx:idx+self.training_args.per_device_eval_batch_size]
            batch_data = np.array(batch_data)
            yield list(batch_data[:,0]), list(batch_data[:,1]) #input, label


    def eval(self, sep="<extra_id_0>"):
        print('Building datasets')
        eval_path = os.path.join(self.dataset_dir,self.test_fn)
        # test_data = process_CONVEX(eval_path)
        pred_rels = json.load(open(self.rel_fn))
        test_data = process_CONVEX_rel(eval_path, pred_rels)
        
        print('evaluating')
        rewrites, pred_rws = [], []
        
        for batch_contexts,batch_rewrites in self.next_batch_data(test_data):
            input_encoding = self.tokenizer(batch_contexts,
                                    padding='longest',
                                    max_length=self.max_in_len,
                                    truncation=True,
                                    return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids=input_encoding['input_ids'],
                                        attention_mask=input_encoding['attention_mask'],
                                        max_length=self.max_out_len,
                                        do_sample=False,
                                        num_beams=10,
                                        num_return_sequences=self.max_candidates)
            batch_answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            rewrites += batch_rewrites
            pred_rws += batch_answers
        
        assert len(rewrites) == len(pred_rws)
        
        bleu4_, bleu3_, bleu2_, bleu1_ = 0.0, 0.0, 0.0, 0.0
        for idx,(ground,pred) in enumerate(zip(rewrites, pred_rws)):
            ground = nltk.word_tokenize(ground.strip())
            pred = nltk.word_tokenize(pred.strip())
            bleu4 = sentence_bleu([ground],pred,weights=(0.25, 0.25, 0.25, 0.25)) #bleu4
            bleu4 = round(bleu4*100,4)
            bleu4_ += bleu4
            bleu3 = sentence_bleu([ground],pred,weights=(0.33, 0.33, 0.33, 0))#bleu3
            bleu3 = round(bleu3*100,4)
            bleu3_ += bleu3
            bleu2 = sentence_bleu([ground],pred,weights=(0.5, 0.5, 0, 0))#bleu2
            bleu2 = round(bleu2*100,4)
            bleu2_ += bleu2
            bleu1 = sentence_bleu([ground],pred,weights=(1.0, 0, 0, 0))#bleu1
            bleu1 = round(bleu1*100,4)
            bleu1_ += bleu1
        avg_bleu4 = bleu4_ / idx
        avg_bleu3 = bleu3_ / idx
        avg_bleu2 = bleu2_ / idx
        avg_bleu1 = bleu1_ / idx
        print("BLEU-4: ", avg_bleu4)
        print("BLEU-3: ", avg_bleu3)
        print("BLEU-2: ", avg_bleu2)
        print("BLEU-1: ", avg_bleu1)
        

if __name__ == '__main__':
    config_fn = 'config_kopl.json'
    with open(config_fn) as f:
        args = json.load(f)
    cqr = CQR(args)
    if args['train_args']['do_train']:
        cqr.train()
    elif args['train_args']['do_eval']:
        cqr.eval()
    
    
    