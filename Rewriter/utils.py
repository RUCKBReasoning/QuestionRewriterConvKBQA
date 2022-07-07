import os
import re
import sys
from sparqlretriever import SparqlRetriever

RELATION_PATTERN      = re.compile('P[0-9\-]+')
ENTITY_PATTERN        = re.compile('Q[0-9]+')
const_interaction_dic = '(and|or)'


class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


def is_date(date):
	pattern = re.compile('^[0-9]+ [A-z]+ [0-9][0-9][0-9][0-9]$')
	if not(pattern.match(date.strip())):
		return False
	else:
		return True


def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    if not(pattern.match(timestamp)):
        return False
    else:
        return True


def convertTimestamp( timestamp):
    yearPattern = re.compile('^[0-9][0-9][0-9][0-9]-00-00T00:00:00Z')
    monthPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-00T00:00:00Z')
    dayPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
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


def convertMonth( month):
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


def formatAnswer(answer):
    if len(answer) == 0:
        return answer
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


def load_dict(filename):
    word2id = dict()
    with open(filename,"r") as f_in:
        for line in f_in:
            word = line.rstrip()
            word2id[word] = len(word2id)
    return word2id


def load_sparql_retriever(sparql_dir):
    sparql_retriever = SparqlRetriever()
    sparql_retriever.load_cache('%s/M2N.json' % sparql_dir,
                               '%s/STATEMENTS.json' % sparql_dir,
                               '%s/QUERY.json' % sparql_dir,
                               '%s/TYPE.json' % sparql_dir,
                               '%s/OUTDEGREE.json' % sparql_dir)
    return sparql_retriever