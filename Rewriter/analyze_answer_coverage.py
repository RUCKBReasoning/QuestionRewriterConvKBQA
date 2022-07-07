import re
import json
from sparqlretriever import SparqlRetriever

sparql_dir = 'KB-cache/'
sparql_retriever = SparqlRetriever()
sparql_retriever.load_cache('%s/M2N.json' % sparql_dir,
                            '%s/STATEMENTS.json' % sparql_dir,
                            '%s/QUERY.json' % sparql_dir,
                            '%s/TYPE.json' % sparql_dir,
                            '%s/OUTDEGREE.json' % sparql_dir)

train_path = 'Datasets/ConvQuestions/train_set/train_simple.json'
dev_path = 'Datasets/ConvQuestions/dev_set/dev_simple.json'
test_path = 'Datasets/ConvQuestions/test_set/test_simple.json'
entity_path = 'Datasets/ConvQuestions/entities.txt'

max_subgraph, total_subgraph, bg1000, total, hit = 0, 0, 0, 0, 0

def is_date(date):
	pattern = re.compile('^[0-9]+ [A-z]+ [0-9][0-9][0-9][0-9]$')
	if not(pattern.match(date.strip())):
		return False
	else:
		return True
def is_year(year):
	pattern = re.compile('^[0-9][0-9][0-9][0-9]$')
	if not(pattern.match(year.strip())):
		return False
	else:
		return True

def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id
entity2id = load_dict(entity_path)
id2entity = {idx: entity for entity, idx in entity2id.items()}


def te_text_in_q(te_texts, q):
    for te in te_texts:
        if te.lower() in q.lower():
            return True
    
    return False

for path in [train_path, dev_path, test_path]:
    with open(path,"r") as f:
        for idx, line in enumerate(f):
            if (idx)%5 != 0:continue
            d = json.loads(line)
            sl = len(d["subgraph"]["tuples"])
            total_subgraph += sl
            if sl>max_subgraph:
                max_subgraph = sl
            if sl > 6000:
                bg1000 += 1
            entities = d["subgraph"]["entities"]
            
            flag = 0
            te_texts = [sparql_retriever.wikidata_id_to_label(id2entity[t]) for t in d['entities']]

            q = d['question']
            for a in d['answers']:
                if entity2id[a['kb_id']] in entities and te_text_in_q(te_texts, q):
                #if entity2id[a['kb_id']] in entities :
                    hit += 1
                    flag = 1
                    break 
            total += 1

print(max_subgraph)
print(total_subgraph/total)
print(bg1000)
print(total)
print(hit)
print(hit/total)
            
