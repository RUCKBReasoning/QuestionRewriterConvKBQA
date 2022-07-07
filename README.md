# Requirements
+ Ubuntu 18.04
+ Python 3.8.12
+ Pytorch 1.11.0

Download the code and set up development environment:
```
pip install -r requirements.txt
```

# Overview

The repository is organized as follows:
+ `KB-cache` KB cache files
+ `BLINK` entity linking tool
+ `models` the pretrained models, indices, and entity embeddings for entity linking
+ `Datasets`
	+ `CANARD` open-domain conversational QA dataset
	+ `ConvQuestions` conversatioanl KBQA dataset
+ `config` config files for training and evaluating
+ `Rewriter` implementation of  question rewriter
+ `Reasoner`
	+ `NSM`  implementation of retrieval-based NSM reasoner
	+ `KoPL`  implementation of semantic parsing-based KoPL reasoner



# Setups
## KB
We download and adopt the KB cache collected by  [Focal Entity](https://github.com/lanyunshi/ConversationalKBQA) in our experiments to save time. You can download [here](https://drive.google.com/drive/folders/1sV-YZanhu80REi2a9bu9Vr-jXziPawXn?usp=sharing) and put the KB-cache directory inside the root directory.
For the KB facts they don't explore, we query the [Wikidata API](https://query.wikidata.org/). You can simply run:
```
python Rewriter/sparqlretriever.py
```
to check whether the Wikidata API is working or not. Check whether the results are:
```
{'head': {'vars': ['r', 'e1']}, 'results': {'bindings': [{'e1': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q7985008'}, 'r': {'type': 'uri', 'value': 'http://www.wikidata.org/prop/direct/P175'}}]}}
```

## Entity Linking
We use ELQ as our entity linking tool. First, clone the BLINK repo as follows:
```
git clone https://github.com/facebookresearch/BLINK.git
```
Place the BLINK directory inside the root directory and follow the setup steps [here](https://github.com/facebookresearch/BLINK/tree/master/elq) to prepare entity linking environment(the directory "models" will be created during the set up process).


## Word Embedding
We employ GloVe as our initialized word embeedings and download the pre-trained word vectors for Wikipedia. You can download [here](https://nlp.stanford.edu/projects/glove/) and put them into Datasets/ConvQuestions/.  Rename the vocabulary file and word embedding file as "vocab_new.txt" and "word_embed_300d.npy".

# Datasets

## ConvQA dataset
We use CANARD dataset to pre-train the question rewriter. You can download [here](https://sites.google.com/view/qanta/projects/canard) and put it into Datasets/.

## ConKBQA dataset
We evaluate our method on the benchmark ConvQuestions. You can download [here](https://convex.mpi-inf.mpg.de/) and put it into Datasets/.

# Rewriter

## Relation Retriever
1. How to construct pseudo (question, relation) dataset?
```
python  Rewriter/retrieve_relation.py --construct
```

2. How to train the relation retriever?
```
python  Rewriter/train_relation_retriever.py
```

## Question Rewriter
1. How to pre-train the question rewriter?
```
python  Rewriter/train_rewriter.py --pre_train
```

2. How to produce pseudo labels for self-training?
```
python  Rewriter/train_rewriter.py --pretrain_generate
python  Rewriter/retriever_topic__entity.py --pre_train
python  Rewriter/retrieve_subgraph.py --pre_train
python  Rewriter/retrieve_relation.py --infer
python  Rewriter/generate_selftrain_datset.py
```

3. How to self-train the question rewriter?
```
python  Rewriter/train_rewriter.py --self_train
```

4. How to generate self-contained  rewritten questions?
```
python  Rewriter/train_rewriter.py --selftrain_generate
```

# Reasoner
## NSM
1. Prepare environment for [NSM](https://github.com/RichardHGL/WSDM2021_NSM).
Prepare Question Rewriter for NSM:
	+ copy the directory "models" into "Reasoner/NSM/" for entity linking
	+ copy the self-trained rewriter model "t5_selftrain_rr" into "Reasoner/NSM/QuestionRewrite" for question rewriting
	+ copy the relaton retriever model "bert_finetune" into "Reasoner/NSM/QuestionRewrite" for relation retrieval

2. How to prepare NSM dataset?
```
python  Rewriter/retriever_topic__entity.py --self_train
python  Rewriter/retrieve_subgraph.py --self_train
python  Rewriter/generate_nsm_dataset.py
```
Execute in the Datsets/ConvQuestions directory:
```
cp entities.txt relations.txt vocab_new.txt word_emb_300d.npy train_set/train_simple.json dev_set/dev_simple.json test_set/test_simple.json ../../Reasoner/NSM/ConvQuestions
```
Execute in the Reasoner/NSM/preprocessing/parse directory:
```
change the path of files in const_parse.sh and dependecy_parse.sh
bash run.sh
```

3. How to train NSM?
Execute in the Reasoner/NSM directory:
```
bash run_ConvQuestions.sh
```

4. How to evaluate Question Rewriter combined with NSM?
Execute in the Reasoner/NSM directory:
```
bash test_ConvQuestions.sh
```

## KoPL
1. Download pre-trained models and KBs for KoPL.
Organize them as follows:
	+ `KoPL`
		+ `KB` KB files
			+ `item.json`
			+ `kb.json`
			+ `wikidata.json`
		+ `Question2KoPL` pre-trained model
			+ `config.json`
			+ `merges.txt`
			+ `pytorch_model.bin`
			+ `training_args.bin`
			+ `vocab.json`

2. How to generate pseduo labels for KoPL?
Execute in the Reasoner/KoPL/code directory and change the file paths to your local paths:
```
python  infer_ConvQuestions.py --pretrain
```

3. How to self-train KoPL?
Execute in the Reasoner/KoPL/code directory:
```
python finetune_kopl.py
```

4. How to evaluate Question Rewriter combined with KoPL?
```
python Rewriter/test_kopl.py
```

## Relation Retriever
 How to evaluate Question Rewriter combined with Relation Retriever?
 ```
 python  Rewriter/retrieve_relation.py --eval
 ```
