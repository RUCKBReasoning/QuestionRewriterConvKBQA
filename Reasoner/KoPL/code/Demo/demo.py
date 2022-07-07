# from flask import Flask
# from flask import request
# from flask import jsonify
# from flask_cors import CORS, cross_origin


import os
import torch
import torch.optim as optim
import torch.nn as nn

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
from kopl.kopl import KoPLEngine

warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

# logging.basicConfig(level=logging.INFO, filemode = 'a', format='%(asctime)s %(levelname)-8s %(message)s')
# logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
# rootLogger = logging.getLogger()
# fin = open('QueryLog/log.txt', 'a')
# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

program_tokenizer = tokenizer_class.from_pretrained('../Question2KoPL')
program_model = model_class.from_pretrained('../Question2KoPL')
program_model = program_model.to(device)
print('loading kb')
# kb = json.load(open('/data1/csl/service/EastQA/Python/kb.json'))
kb = json.load(open('../KB/kb.json'))
print('kb loaded')
program_rule_executor = KoPLEngine(kb)





def program_vis(text):
    pattern = re.compile(r'(.*?)\((.*?)\)')
    def get_dep(program, inputs):
        # logging.info(program)
        # logging.info(inputs)
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
        # logging.info(program)
        # logging.info(inputs)
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
            max_length = 500,
        )
        outputs = [program_tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs]
        output = outputs[0]
        # filter_pattern = re.compile("FilterConcept(.*?)<b>")
        # output = filter_pattern.sub("",output,)
        print(output)
        chunks = output.split('<b>')
        func_list = []
        inputs_list = []
        for chunk in chunks:
            # logging.info(chunk)
            res = pattern.findall(chunk)
            # logging.info(res)
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
    return program, func_list, inputs_list


import random

# @app.route('/', methods=['POST', 'GET'])

# @cross_origin()
def main():
    start_time = time.time()
    #question = request.form['question'].strip()
    question = "Who is Argentina's youngest goal keeper?"
    program, func_list, inputs_list = program_vis(question)
    program_result = program_rule_executor.forward(func_list, inputs_list, ignore_error = True, show_details = False)
    print(program_result)
    if isinstance(program_result, list) and len(program_result) > 0:
        program_result = program_result[0]
    if program_result is None or program_result == 'None' or program_result == []:
        program_result = 'Not found in our current KB'
    end_time = time.time()
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    #logging.info(json.dumps({'timestamp': time_, 'question': question, 'program': program, 'program_result': program_result, 'time': end_time - start_time}, indent = 1))
    #fin.write(json.dumps({'timestamp': time_, 'question': question, 'program': program, 'program_result': program_result, 'time': end_time - start_time}) + '\n')
    #fin.flush()
    print({'program': program, 'program_result': program_result, 'time': end_time - start_time})
    return {'program': program, 'program_result': program_result, 'time': end_time - start_time}


if __name__ == '__main__':
    #app.run(host = '0.0.0.0', port = 6058, threaded=True)
    main()
