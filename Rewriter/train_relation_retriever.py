
from collections import defaultdict
import os
import sys
import math
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def load_pretrained_model(name):
    if name == 'roberta-base':
        model = RobertaModel.from_pretrained('roberta-base')
        hdim = 768
    elif name == 'roberta-large':
        model = RobertaModel.from_pretrained('roberta-large')
        hdim = 1024
    elif name == 'bert-large':
        model = BertModel.from_pretrained('bert-large-uncased')
        hdim = 1024
    else: #bert base
        model = BertModel.from_pretrained('bert-base-uncased')
        hdim = 768
    return model, hdim


def load_tokenizer(name):
    if name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif name == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    elif name == 'bert-large':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    else: #bert base
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


class QuestionEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_question):
        super(QuestionEncoder, self).__init__()

        #load pretrained model as base for question encoder and KB information encoder
        
        self.question_encoder, self.question_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_question

    def forward(self, input_ids, attn_mask):
        #encode question text
        if self.is_frozen:
            with torch.no_grad(): 
                question_output = self.question_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            question_output = self.question_encoder(input_ids, attention_mask=attn_mask)[0]
        #training model to put all sense information on CLS token 
        question_output = question_output[:,0,:].squeeze(dim=1) #now bsz*question_hdim
        
        return question_output


class RelationEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_relation):
        super(RelationEncoder, self).__init__()

        #load pretrained model as base for question encoder and KB information encoder
        self.relation_encoder, self.relation_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_relation

    def forward(self, input_ids, attn_mask):
        #encode kb information
        if self.is_frozen:
            with torch.no_grad(): 
                relation_output = self.relation_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            relation_output = self.relation_encoder(input_ids, attention_mask=attn_mask)[0]

        #average representations over target word(s)
        relation_output = relation_output[:,0,:].squeeze(dim=1) #num*relation_hdim,num=te_num*rel_num
        #relation_output = torch.mean(relation_output, dim=1) #num*relation_hdim,num=te_num*rel_num

        return relation_output


class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_question=False, freeze_relation=False):
        super(BiEncoderModel, self).__init__()

        #load pretrained model as base for question encoder and KB information encoder
        self.question_encoder = QuestionEncoder(encoder_name, freeze_question)
        self.relation_encoder = RelationEncoder(encoder_name, freeze_relation)
        assert self.question_encoder.question_hdim == self.relation_encoder.relation_hdim

    def question_forward(self, question_input, question_input_mask):
        return self.question_encoder.forward(question_input, question_input_mask)

    def relation_forward(self, relation_input, relation_mask):
        return self.relation_encoder.forward(relation_input, relation_mask)


class CrossEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_encoder=False, hidden_dropout_prob=0.1):
        super(CrossEncoderModel, self).__init__()

        self.cross_encoder, self.encoder_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_encoder
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.scorer = torch.nn.Linear(self.encoder_hdim, 1)
        

    def forward(self, input_ids, attn_mask):
        if self.is_frozen:
            with torch.no_grad(): 
                output = self.cross_encoder(input_ids, attention_mask=attn_mask, return_dict=True)[0]
        else:
            output = self.cross_encoder(input_ids, attention_mask=attn_mask, return_dict=True)[0]

        output = output[:,0,:].squeeze(dim=1) 
        output = self.dropout(output)
        pred_scores = self.scorer(output)

        return pred_scores 


class RRDataset(Dataset):
    def __init__(self, data):
        self.data      = data
        self.data_size = len(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class RelationRetriever():
    def __init__(self, args) -> None:
        self.data_dir     = args["dataset_dir"]
        self.train_fn     = os.path.join(self.data_dir, args["train_fn"])
        self.dev_fn       = os.path.join(self.data_dir, args["valid_fn"])
        self.test_fn      = os.path.join(self.data_dir, args["test_fn"])
        self.output_dir   = args["output_dir"]
        self.max_len      = args["max_len"]
        self.neg_num      = args["neg_sample_num"]
        self.pos_num      = args["pos_trunc_num"]
        self.epochs       = args["train_epoches"]
        self.train_bsz    = args["train_batch_size"]
        self.eval_bsz     = args["eval_batch_size"]
        self.weight_decay = args['weight_decay']
        self.grad_norm    = args["max_grad_norm"]
        
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)
        
        print("loading datasets...")
        self.train_dataset    = self.build_dataset(self.train_fn)
        self.dev_dataset      = self.build_dataset(self.dev_fn)
        self.test_dataset     = self.build_dataset(self.test_fn)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.train_bsz, shuffle=True,  drop_last=False, collate_fn=self.train_data_collator)
        self.dev_dataloader   = DataLoader(dataset=self.dev_dataset,   batch_size=self.eval_bsz,  shuffle=True,  drop_last=False, collate_fn=self.train_data_collator)
        self.test_dataloader  = DataLoader(dataset=self.test_dataset,  batch_size=self.eval_bsz,  shuffle=False, drop_last=False, collate_fn=self.eval_data_collator)
        print("datasize for train/valid/test: ", self.train_dataset.data_size,self.dev_dataset.data_size,self.test_dataset.data_size)
        
        
        print("setting up model and optimizer...")
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model     = CrossEncoderModel(args['encoder_name'],
                                           freeze_encoder=args['freeze_encoder'], 
                                           hidden_dropout_prob=args["hidden_dropout_prob"]).to(self.device)
        self.tokenizer = load_tokenizer(args['encoder_name'])
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        adam_epsilon   = 1e-8
        total_steps    = self.train_dataset.data_size*self.epochs
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=adam_epsilon)
        self.schedule  = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps = total_steps)
    
    
    def save_model(self):
        fpath = os.path.join(self.output_dir, 'best_model.pt')
        torch.save(self.model.state_dict(), fpath)
    
    def load_model(self):
        fpath = os.path.join(self.output_dir, 'best_model.pt')
        loaded_paras = torch.load(fpath)
        self.model.load_state_dict(loaded_paras)
    
    def build_dataset(self, fn):
        all_data = [] 
        with open(fn,"r") as f: 
            for line in f:
                data = json.loads(line)
                process_data = {'pos':[[data['question'], pos_r] for pos_r in data['pos_r']], 'neg':[[data['question'], neg_r] for neg_r in data['neg_r']]}
                if len(process_data['pos'])>self.pos_num:continue
                all_data.append(process_data)
        return RRDataset(all_data)

    
    def train_data_collator(self, batch):
        # fix num, multi-positive
        max_num = self.neg_num
        batch_data, batch_labels = [], []
        for data in batch:
            data_num = len(data['pos'])+len(data['neg'])
            if data_num < max_num:
                pad_num = max_num - data_num
                if pad_num<=len(data['neg']):
                    data['neg'] += list(random.sample(data['neg'], pad_num))
                else:
                    data['neg'] += data['neg']*int(pad_num/len(data['neg'])) + list(random.sample(data['neg'], (pad_num%len(data['neg']))))
            else:
                neg_num = max_num - len(data['pos'])
                assert neg_num>0
                data['neg'] = list(random.sample(data['neg'], neg_num))
            label = [1]*len(data['pos'])+[0]*len(data['neg'])
            assert len(label)==max_num
            batch_labels.append(label)
            batch_data += (data['pos']+data['neg'])

        assert len(batch_labels) == len(batch)
        assert len(batch_data) == len(batch)*max_num
        encodings = self.tokenizer([d[0] for d in batch_data],
                                    [d[1] for d in batch_data],
                                    return_tensors="pt",
                                    padding=True,
                                    truncation='longest_first',
                                    max_length=self.max_len).to(self.device)
        
        return max_num, encodings, torch.FloatTensor(batch_labels).to(self.device) #input, label
    
    '''
    def data_collator(self, batch):
        # fix num, single positive
        max_num = self.neg_num
        batch_data, batch_labels = [], []
        for data in batch:
            #print(len(data['pos']), data['pos'][0][0])
            for pos_r in data['pos']:
                data_num = 1+len(data['neg'])
                if data_num < max_num:
                    pad_num = max_num - data_num
                    if pad_num<=len(data['neg']):
                        data['neg'] += list(random.sample(data['neg'], pad_num))
                    else:
                        data['neg'] += data['neg']*int(pad_num/len(data['neg'])) + list(random.sample(data['neg'], (pad_num%len(data['neg']))))
                else:
                    neg_num = max_num - 1
                    assert neg_num>0
                    data['neg'] = list(random.sample(data['neg'], neg_num))
                label = [1]+[0]*len(data['neg'])
                assert len(label)==max_num
                batch_labels.append(label)
                batch_data += ([pos_r]+data['neg'])

        assert len(batch_data) == len(batch_labels)*max_num
        encodings = self.tokenizer([d[0] for d in batch_data],
                                    [d[1] for d in batch_data],
                                    return_tensors="pt",
                                    padding=True,
                                    truncation='longest_first',
                                    max_length=self.max_len).to(self.device)
        
        return max_num, encodings, torch.FloatTensor(batch_labels).to(self.device) #input, label
    '''
    
    def eval_data_collator(self, batch):
        max_num = max([len(data['pos'])+len(data['neg']) for data in batch])
        batch_data, batch_labels = [], []
        for data in batch:
            pad_num = max_num - (len(data['pos'])+len(data['neg']))
            if pad_num<=len(data['neg']):
                data['neg'] += list(random.sample(data['neg'], pad_num))
            else:
                data['neg'] += data['neg']*int(pad_num/len(data['neg'])) + list(random.sample(data['neg'], (pad_num%len(data['neg']))))
            label = [1]*len(data['pos'])+[0]*len(data['neg'])
            assert len(label)==max_num
            batch_labels.append(label)
            batch_data += (data['pos']+data['neg'])
        assert len(batch_labels) == len(batch)
        assert len(batch_data) == len(batch)*max_num
        encodings = self.tokenizer([d[0] for d in batch_data],
                                    [d[1] for d in batch_data],
                                    return_tensors="pt",
                                    padding=True,
                                    truncation='longest_first',
                                    max_length=self.max_len).to(self.device)
        
        return max_num, encodings, torch.FloatTensor(batch_labels).to(self.device) #input, label
    

    def train_retriever(self):
        print('start training...')
        sys.stdout.flush()
        self.model.train()
        
        best_dev_h1 = 0.
        for epoch in range(1, self.epochs+1):
            idx = 0
            for batch_data in tqdm(self.train_dataloader):
                batch_loss = 0.
                max_num, data, label = batch_data
                #print(max_num)
                data_size = label.size(0)
                pred_scores = self.model(data['input_ids'], data['attention_mask'])
                pred_scores = pred_scores.view(data_size, -1)
                assert pred_scores.size(0) == label.size(0)
                assert pred_scores.size(1) == label.size(1)
                loss = self.criterion(pred_scores, label)
                loss.backward()
                
                # batch_loss += loss.item()
                # if idx!=0 and idx%10==0:
                #     print("loss: ", batch_loss/10)    

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()
                self.schedule.step() # Update learning rate schedule
                self.model.zero_grad()
                
                idx += 1
            
            with torch.no_grad():
                dev_h1,_,_ = self.eval_retriever(mode='dev')
                print("dev h1: ", dev_h1)
            if dev_h1 >= best_dev_h1:
                print('updating best model at epoch {}...'.format(epoch))
                sys.stdout.flush() 
                best_dev_h1 = dev_h1
                self.save_model()
        return
    
    
    def eval_retriever(self, mode='test'):
        print('start evaluating...')
        
        if mode=='test':
            self.load_model()
            
        sys.stdout.flush()
        self.model.eval()
        if mode=='dev':
            eval_dataloader = self.dev_dataloader
            total = self.dev_dataset.data_size
        elif mode=='test':
            eval_dataloader = self.test_dataloader
            total = self.test_dataset.data_size

        hit1,hit3,hit5 = 0,0,0
        for batch_data in tqdm(eval_dataloader):
            max_num, data, label = batch_data
            data_size = label.size(0)
            pred_scores = self.model(data['input_ids'], data['attention_mask'])
            pred_scores = pred_scores.view(data_size, -1)
            #print(pred_scores)
            for i,j in enumerate(torch.argmax(pred_scores, dim=1).cpu().tolist()):
                hit1 += int(label[i][j].cpu())
            
            for i,top3 in enumerate(torch.topk(pred_scores, 3, dim=1, largest=True, sorted=True)[1]):
                for j in top3:
                    if int(label[i][j].cpu())==1:
                        hit3 += 1
                        break
            
            for i,top5 in enumerate(torch.topk(pred_scores, 5, dim=1, largest=True, sorted=True)[1]):
                for j in top5:
                    if int(label[i][j].cpu())==1:
                        hit5 += 1
                        break
         
        return (hit1/total),(hit3/total),(hit5/total)

    
    def get_batch_infer_data(self, all_data, batch_size):
        for idx in range(0,len(all_data),batch_size):
            batch = all_data[idx:idx+batch_size]
            max_num = max([len(data) for data in batch])
            batch_data = []
            for data in batch:
                pad_num = max_num - len(data)
                if pad_num<=len(data):
                    data += list(random.sample(data, pad_num))
                else:
                    data += data*int(pad_num/len(data)) + list(random.sample(data, (pad_num%len(data))))
                batch_data += data
            assert len(batch_data) == len(batch)*max_num
            encodings = self.tokenizer([d[0] for d in batch_data],
                                       [d[1] for d in batch_data],
                                       return_tensors="pt",
                                       padding=True,
                                       truncation='longest_first',
                                       max_length=self.max_len).to(self.device)
            
            yield max_num, encodings, batch_data
    
    
    def infer_retriever(self, eval_data, mode='test'):
        '''
        :format eval_data: dict{k1 : [ [[q1,r11],[q1,r12],...]], k2 : [[q2,r21],[q2,r22],...] ...]}
        '''
        # print('start evaluating...')
        # sys.stdout.flush()

        #self.load_model()
        #self.model.eval()
        with torch.no_grad():
            pred_rels = defaultdict(list)
            for k,data in eval_data.items():
                encodings = self.tokenizer([d[0] for d in data],
                                        [d[1] for d in data],
                                        return_tensors="pt",
                                        padding=True,
                                        truncation='longest_first',
                                        max_length=self.max_len).to(self.device)
                
                pred_scores = self.model(encodings['input_ids'], encodings['attention_mask'])
                pred_scores = pred_scores.squeeze(dim=1)
                top5 = torch.topk(pred_scores, 5, dim=0, largest=True, sorted=True)[1]
                pred_rels[k]= [data[j][1] for j in top5]

        assert len(pred_rels) == len(eval_data)
        return pred_rels


if __name__ == "__main__":
	if not torch.cuda.is_available():
		print("Need available GPU(s) to run this model...")
		quit()

	config_fn = 'config/config_relation_retriever.json'
	with open(config_fn) as f:
		args = json.load(f)

	#set random seeds
	torch.manual_seed(args['rand_seed'])
	os.environ['PYTHONHASHSEED'] = str(args['rand_seed'])
	torch.cuda.manual_seed(args['rand_seed'])
	torch.cuda.manual_seed_all(args['rand_seed'])   
	np.random.seed(args['rand_seed'])
	random.seed(args['rand_seed'])
	
	relation_retriever = RelationRetriever(args)
	if args['do_train']:
		relation_retriever.train_retriever()
	elif args['do_eval']:
		print(relation_retriever.eval_retriever())