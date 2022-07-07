
import os
import re
import json
import requests
import time, datetime
import urllib
import urllib.request  as urllib2
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON

AGENT = 'Chrome/77.0.3865.90'

def test():
    if True:
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=AGENT) #    PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>
        sparql_txt = """
        SELECT DISTINCT ?r ?e1 WHERE {
        FILTER (EXISTS {?e1 rdfs:label ?name.} || datatype(?e1) in (xsd:dateTime))
        ?e1 ?r wd:Q3127867}
        """
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        if True:
            results = sparql.query().convert()
            print(results)
        # except urllib2.HTTPError as err:
        #     if err.code == 403:
        #         print('results')
        # except:
        #     print('QueryBadFormed')
    elif False:
        sparql_txt = """select ?e0 WHERE {
          wd:Q63885185 rdfs:label u'Danny Zuko'.
        }
                """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        requests.post('https://query.wikidata.org/sparql', headers = headers, data = data)
    else:
        print('Your database is not installed properly !!!')



def convert_json_to_save(dic):
    new_dic = {}
    for hs in dic:
        new_h = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in hs]) if isinstance(hs, tuple) else hs
        if len(dic[hs]) == 0: new_dic[new_h] = {}
        for rs in dic[hs]:
            new_r = '\t'.join([' '.join(r) for r in rs])
            new_t = list(dic[hs][rs])
            if new_h not in new_dic:
                new_dic[new_h] = {new_r: new_t}
            elif new_r not in new_dic[new_h]:
                new_dic[new_h][new_r] = new_t
    return new_dic


def convert_json_to_load(dic):
    new_dic = defaultdict(dict)
    for hs in dic:
        new_h = hs 
        if len(dic[hs]) == 0: new_dic[new_h] = {}
        for rs in dic[hs]:
            new_r = tuple([tuple(r.split(' ')) for r in rs.split('\t')])
            new_t = set(dic[hs][rs])
            if new_h not in new_dic:
                new_dic[new_h] = {new_r: new_t}
            elif new_r not in new_dic[new_h]:
                new_dic[new_h][new_r] = new_t
    return new_dic


def get_delay(date):
    try:
        date = datetime.datetime.strptime(date, '%a, %d %b %Y %H:%M:%S GMT')
        timeout = int((date - datetime.datetime.now()).total_seconds())
    except ValueError:
        timeout = int(date)
    return timeout


def form_trips(p_tokens):
    t_idx, d_idx = 0, 0
    trips = ()
    for p_token in p_tokens:
        trip = ()
        if len(p_token) > 3: p_token = p_token[1:]
        for p_idx, e in enumerate(p_token):
            e = 'wd:%s' %e if (re.search('^Q', e) or len(e.split('.')) > 2) else "'%s'^^xsd:date" %e if (e.isdigit() and int(e) < 2100 or re.search('\d-\d', e)) else e
            if re.search('^\?e', e): t_idx = int(re.findall('\d+', e)[0])
            if re.search('^\?d', e): d_idx = int(re.findall('\d+', e)[0])
            trip += (e, )
        trips += (' '.join(trip), )
    if d_idx < t_idx - 1: d_idx = 0 # if d is too far, unvalid it
    return trips, t_idx, d_idx


class SparqlRetriever(object):
    '''A class for sparql retrival'''

    def __init__(self):
        self.SPARQLPATH = "https://query.wikidata.org/sparql"
        
        #cache file
        self.M2N        = {}
        self.STATEMENTS = defaultdict(dict) 
        self.QUERY_TXT  = set()
        self.TYPE       = {}
        self.OUTDEGREE  = {}
        

    def load_cache(self, M2N_file, STATEMENTS_file, QUERY_file, TYPE_file, OUTDEGREE_file):
        self.M2N_file        = M2N_file
        self.STATEMENTS_file = STATEMENTS_file
        self.QUERY_file      = QUERY_file
        self.TYPE_file       = TYPE_file
        self.OUTDEGREE_file  = OUTDEGREE_file

        if os.path.exists(M2N_file):
            self.M2N = json.load(open(M2N_file))
        if os.path.exists(STATEMENTS_file):
            STATEMENTS = json.load(open(STATEMENTS_file))
            self.STATEMENTS = convert_json_to_load(STATEMENTS)
        if os.path.exists(QUERY_file):
            self.QUERY_TXT = set(json.load(open(QUERY_file)))
        if os.path.exists(OUTDEGREE_file):
            self.OUTDEGREE = json.load(open(OUTDEGREE_file))
        if os.path.exists(TYPE_file):
            self.TYPE = json.load(open(TYPE_file))


    def save_cache(self):
        json.dump(self.M2N, open(self.M2N_file, 'w'))
        json.dump(convert_json_to_save(self.STATEMENTS), open(self.STATEMENTS_file, 'w'))
        json.dump(list(self.QUERY_TXT), open(self.QUERY_file, 'w'))
        json.dump(self.TYPE, open(self.TYPE_file, 'w'))
        json.dump(self.OUTDEGREE, open(self.OUTDEGREE_file, 'w'))


    def SQL_description(self, qid):
        sparql_txt="""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX schema: <http://schema.org/>

        SELECT ?o
        WHERE 
        {
            """ 'wd:' + qid +' schema:description ?o.FILTER ( lang(?o) = "en" )' """
        }
        """
        sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        rtn = set()
        try:
            results = sparql.query().convert()
            if results['results']['bindings']:
                for t in results['results']['bindings']:
                    rtn.add((t['o']['value']))
        except urllib2.HTTPError as err:
            if err.code == 429:
                time.sleep(60)
        except:
            pass
        
        return rtn


    def SQL_string_entities(self, qid):
        relation_prefix = 'http://www.wikidata.org/prop/direct/'
        sparql_txt="""
        PREFIX wd: <http://www.wikidata.org/entity/>

        SELECT ?r ?o
        WHERE 
        {
            """ 'wd:' + qid +' ?r ?o. FILTER ( lang(?o) = "en" )' """
        }
        """
        sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        
        rtn = set()
        try:
            results = sparql.query().convert()
            if results['results']['bindings']:
                for t in results['results']['bindings']:
                    if t['r']['value'].startswith(relation_prefix):
                        r = t['r']['value'].split('/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                        e = t['o']['value'].split('/entity/')[-1] if re.search('^http', t['o']['value']) else t['o']['value']
                        rtn.add((r,e))
        except urllib2.HTTPError as err:
            if err.code == 429:
                time.sleep(60)
        except:
            pass
        
        return rtn


    def SQL_1hop(self, p, QUERY=None):
        '''
        retrieve KB for 1hop neighbors
        :param p : tuple of topic entity id, eg: ((t,),)
        :param QUERY : cached SPARQL queries
        '''
        kbs, sparql_txts = defaultdict(set), set()

        trips, t_idx, _ = form_trips(p)
        topic = re.findall('wd\:Q[0-9]+', trips[0])[0]
        queries = [(' '.join(['%s' %topic, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?e%s' %(t_idx+1), '?r', '%s' %topic]), )] if t_idx == 0 else (' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )
        retu = ' '.join(['?r', '?e%s' %(t_idx+1)])
        const = "FILTER (EXISTS {?e%s rdfs:label ?name.} || datatype(?e%s) in (xsd:dateTime) || isNumeric(?e%s))" %(t_idx+1, t_idx+1, t_idx+1)
        const1 = ""

        for query_idx, query in enumerate(queries):
            #If const1_idx > 1, consider the name-mentioned relation
            
            trips = '.\n'.join(query) if t_idx == 0 else '.\n'.join(trips + query) #LIMIT 500
            sparql_txt = """SELECT DISTINCT %s WHERE {%s\n%s\n%s}""" %(retu, const, const1, trips)
            sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            try:
                if (QUERY is not None) and sparql_txt in QUERY:
                    continue #return kbs, sparql_txts
                
                results = sparql.query().convert()
                if results['results']['bindings']:
                    for t in results['results']['bindings']:
                        r = t['r']['value'].split('/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                        t = t['e%s' %(t_idx+1)]['value'].split('/entity/')[-1] if re.search('^http', t['e%s' %(t_idx+1)]['value']) else t['e%s' %(t_idx+1)]['value']
                        if query_idx == 0:
                            trip = ((p[0][0], r, '?e%s' %(t_idx+1)), ) if t_idx == 0 else (('?e%s' %t_idx, r, '?e%s' %(t_idx+1)), )
                        elif query_idx == 1:
                            trip = (('?e%s' %(t_idx+1), r, p[0][0]), ) if t_idx == 0 else (('?e%s' %t_idx, r, '?e%s' %(t_idx+1)), )
                        kbs[trip].add(t)
                sparql_txts.add(sparql_txt)
            except urllib2.HTTPError as err:
                if err.code == 429:
                    time.sleep(60)
            except:
                pass
        
        return kbs, sparql_txts


    def SQL_2hop(self, p, QUERY=None):
        ''' Similar to SQL_1hop '''
        kbs, sparql_txts = defaultdict(set), set()

        trips, t_idx, _ = form_trips(p)
        topic = re.findall('wd\:Q[0-9]+', trips[0])[0]
        query = (' '.join(['%s' %topic, '?r', '?d%s' %(t_idx+1)]), ) if t_idx == 0 else (' '.join(['?e%s' %t_idx, '?r', '?d%s' %(t_idx+1)]), )
        query += (' '.join(['?d%s' %(t_idx+1), '?r1', '?e%s' %(t_idx+2)]), )      
        trips = '.\n'.join(query) if t_idx == 0 else '.\n'.join(trips + query)
        retu = ' '.join(['?r', '?r1', '?e%s' %(t_idx+2)])
        const = "FILTER (?e%s!=%s).\nMINUS {?d%s rdfs:label ?name.}." %((t_idx+2, topic, t_idx+1))

        for const1_idx, const1 in enumerate(["?e%s rdfs:label ?name." %(t_idx+2)]): #LIMIT 300
            sparql_txt = """SELECT DISTINCT %s WHERE {%s\n%s\n%s}""" %(retu, const, const1, trips)
            #print('sparql_txt', sparql_txt)
            sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            try:
                if (QUERY is not None) and sparql_txt in QUERY:
                    continue #return kbs, sparql_txts
                
                results = sparql.query().convert()
                #print(len(results['results']['bindings']))
                if results['results']['bindings']:
                    for t in results['results']['bindings']:
                        r = t['r']['value'].split('/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                        r1 = t['r1']['value'].split('/')[-1] if re.search('^http', t['r1']['value']) else t['r1']['value']
                        t = t['e%s' %(t_idx+2)]['value'].split('/entity/')[-1] if re.search('^http', t['e%s' %(t_idx+2)]['value']) else t['e%s' %(t_idx+2)]['value']
                        trip = ((p[0][0], r, '?d%s' %(t_idx+1)), ('?d%s' %(t_idx+1), r1, '?e%s' %(t_idx+2))) if t_idx == 0 else (('?e%s' %t_idx, r, '?d%s' %(t_idx+1)), ('?d%s' %(t_idx+1), r1, '?e%s' %(t_idx+2)))
                        kbs[trip].add(t)
                sparql_txts.add(sparql_txt)
            except urllib2.HTTPError as err:
                if err.code == 429:
                    time.sleep(60)
            except:
                pass

        return kbs, sparql_txts


    def SQL_1hop_interaction(self, p, const_entities, QUERY=None):
        '''
        :param const_entities: pre-defined operations
        const_interaction_dic = '(and|or)'
        '''
        kbs, const1, sparql_txts = defaultdict(set), '', set()
        
        raw_trips, t_idx, d_idx = form_trips(p)
        
        retu = ' '.join(['?e%s' %(t_idx), '?r1', '?r2'])
        const = "?e%s rdfs:label ?name." %t_idx
        const1 = ""
        if list(const_entities)[0] == 'or':
            queries = [(' '.join(['?e%s' % (t_idx), '?r1', raw_trips[0]]), ' '.join(['?e%s' % (t_idx), '?r1', raw_trips[1]]))]
            queries += [(' '.join([raw_trips[0], '?r1', '?e%s' % (t_idx)]), ' '.join([raw_trips[1], '?r1', '?e%s' % (t_idx)]))]
            const_type = 'union'
            trips = [' UNION '.join(['{'+q+'}' for q in query]) for query in queries]      
        elif list(const_entities)[0] == 'and':
            queries = [(' '.join(['?e%s' % (t_idx), '?r1', raw_trips[0]]), ' '.join(['?e%s' % (t_idx), '?r2', raw_trips[1]]))]
            queries += [(' '.join([raw_trips[0], '?r1', '?e%s' % (t_idx)]), ' '.join([raw_trips[1], '?r2', '?e%s' % (t_idx)]))]
            const_type = 'intersection'
            trips = ['.'.join(query) for query in queries]

        for trip in trips:
            sparql_txt = """SELECT DISTINCT %s WHERE {%s.%s}""" %(retu, trip, const)
            
            sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            try:
                results = sparql.query().convert()
                if results['results']['bindings']:
                    for t in results['results']['bindings']:
                        if 'r1' in t:
                            r1 = t['r1']['value'].split('/')[-1] if re.search('^http', t['r1']['value']) else t['r1']['value']
                        if 'r2' in t:
                            r2 = t['r2']['value'].split('/')[-1] if re.search('^http', t['r2']['value']) else t['r2']['value']
                        raw_t = t['e%s' % (t_idx)]['value'].split('/entity/')[-1] if re.search('^http', t['e%s' % (t_idx)]['value']) else t['e%s' % (t_idx)]['value']
                        if ('r2' in t) and ('r1' in t):
                            trip = ((const_type, '?e%s' %(t_idx), r1, raw_trips[0]), ('?e%s' %(t_idx), r2, raw_trips[1]))
                        else:
                            if ('r1' in t):
                                trip = ((const_type, '?e%s' %(t_idx), r1, raw_trips[0]), )
                            elif ('r2' in t):
                                trip = ((const_type, '?e%s' %(t_idx), r2, raw_trips[1]), )
                        kbs[trip].add(raw_t)
                sparql_txts.add(sparql_txt)
            except urllib2.HTTPError as err:
                if err.code == 429:
                    time.sleep(60)
            except:
                pass

        return kbs, sparql_txts


    def SQL_1hop_reverse(self, p, const_entities, QUERY=None):
        '''
        :param const_entities: pre-defined operations
        const_minimax_dic = 'amount|number|how many|final|first|last|predominant|biggest|major
                            |warmest|tallest|current|largest|most|newly|son|daughter'
        const_interaction_dic = '(and|or)'
        const_verification_dic = '(do|is|does|are|did|was|were)'
        '''
        kbs, const1, sparql_txts = defaultdict(set), '', set()

        raw_trips, t_idx, d_idx = form_trips(p)
        if len(raw_trips): raw_trips = ("VALUES ?e0 {%s}" % raw_trips[0],)
        #print('raw_trips', raw_trips)

        if re.search('^Q', list(const_entities)[0]):
            const_type = 'mid'
            const_entiti = ['wd:%s' %e for e in const_entities]
            const = "VALUES ?e%s {%s}" %(t_idx+1, ' '.join(sorted(const_entiti)))
            queries = [(' '.join(['?e%s' %(t_idx+1), '?r', '?e%s' %t_idx]), ), (' '.join(['?e%s' %(t_idx+1), '?r', '?d%s' %d_idx]), )] if d_idx else [(' '.join(['?e%s' %(t_idx+1), '?r', '?e%s' %t_idx]), )]
        elif False: #re.search('^\d', list(const_entities)[0]):
            const_type = 'year'
            year = int(list(const_entities)[0])
            const = "FILTER(Year(?e%s) >= Year('%s'))" %(t_idx+1, year, t_idx+1, year+1)
            queries = [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?d%s' %d_idx, '?r', '?e%s' %(t_idx+1)]), )] if d_idx else [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )]
        elif list(const_entities)[0] in ['first', 'last', 'current', 'newly', 'final']:
            const_type = list(const_entities)[0]
            order = 'ASC' if const_type in ['first'] else 'DESC'
            const1 = '.'.join(['ORDER BY %s (xsd:dateTime(?e%s))\nLIMIT 1' % (order, t_idx + 2)])
            const = '.'.join(['?e%s rdfs:label ?name' % (t_idx + 1),
                              'FILTER (datatype(?e%s) in (xsd:dateTime))' % (t_idx + 2)])
            queries = [('?e%s ?r ?e%s' % (t_idx, t_idx + 1), '?e%s ?r2 ?e%s' % (t_idx + 1, t_idx + 2))]
        elif False: #list(const_entities)[0] in ['largest', 'most', 'predominant', 'biggest', 'major', 'warmest', 'tallest']:
            const_type = list(const_entities)[0]
            const = "VALUES ?r {%s}" %const2rel[const_type]
            const1 = "ORDER BY DESC(xsd:float(?e%s))\nLIMIT 1" %(t_idx+1)
            queries = [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?d%s' %d_idx, '?r', '?e%s' %(t_idx+1)]), )] if d_idx else [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )]
        elif list(const_entities)[0] in ['amount', 'number', 'how many']:
            const_type = 'amount'
            const = "VALUES ?r {wdt:P527 wdt:P166}"
            queries = [(' '.join(['?e%s' % t_idx, '?r', '?e%s' % (t_idx + 1)]),), (' '.join(['?d%s' % d_idx, '?r', '?e%s' % (t_idx + 1)]),)] if d_idx else [(' '.join(['?e%s' % t_idx, '?r', '?e%s' % (t_idx + 1)]),)]
        elif False: #list(const_entities)[0] in ['daughter', 'son']:
            const_type = list(const_entities)[0]
            const = "VALUES ?e%s {ns:m.05zppz}" %(t_idx+1) if const_type in ['son'] else "VALUES ?e%s {ns:m.02zsn}" %(t_idx+1)
            queries = [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?d%s' %d_idx, '?r', '?e%s' %(t_idx+1)]), )] if d_idx else [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )]
        else:
            return kbs, sparql_txts
            #raise Exception('SQL_1hop_reverse has wrong constraint format %s' %str(const_entities))
        
        for q_idx, query in enumerate(queries):
            trips = '.\n'.join(raw_trips + query)
            retu = ' '.join(['?e%s' %(t_idx+1), '?r', '?e%s' %t_idx])
            sparql_txt = """SELECT DISTINCT %s WHERE {%s.\n%s}%s""" %(retu, trips, const, const1)
            #print(sparql_txt)
            sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            try:
                if (QUERY is not None) and sparql_txt in QUERY:
                    continue
                results = sparql.query().convert()
                if results['results']['bindings']:
                    for t in results['results']['bindings']:
                        h = t['e%s' %(t_idx)]['value'].split('/entity/')[-1] if re.search('^http', t['e%s' %(t_idx)]['value']) else t['e%s' %(t_idx)]['value']
                        r = t['r']['value'].split('/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                        t = t['e%s' %(t_idx+1)]['value'].split('/entity/')[-1] if re.search('^http', t['e%s' %(t_idx+1)]['value']) else t['e%s' %(t_idx+1)]['value']
                        if const_type in ['mid']:
                            kbs[(('?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[(('?d%s' %d_idx, r, h), )].add(t)
                        elif const_type in ['son', 'daughter']:
                            kbs[((const_type, '?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[((const_type, '?d%s' %d_idx, r, h), )].add(t)
                        elif const_type in ['year']:
                            kbs[(('?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[(('?d%s' %d_idx, r, h), )].add(t)
                        elif const_type == 'amount':
                            kbs[((const_type, h, r, '?e%s' %t_idx), )].add(str(len(results['results']['bindings']))) if q_idx==0 else kbs[((const_type, h, r, '?d%s' %d_idx), )].add(str(len(results['results']['bindings'])))
                        elif const_type in ['first', 'last', 'current', 'final']:
                            kbs[((const_type, h, r, '?e%s' %t_idx), )].add(t) if q_idx==0 else kbs[((const_type, h, r, '?d%s' %d_idx), )].add(t)
                        elif const_type in ['largest', 'most', 'predominant', 'biggest', 'major', 'warmest', 'tallest']:
                            kbs[((const_type, '?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[((const_type, '?d%s' %d_idx, r, h), )].add(t)
                sparql_txts.add(sparql_txt)
            except urllib2.HTTPError as err:
                if err.code == 429:
                    time.sleep(60)
            except:
                pass

        return kbs, sparql_txts


    def SQL_2hop_reverse(self, p, const_entities, QUERY=None):
        kbs, const1, sparql_txts = defaultdict(set), '', set()

        raw_trips, t_idx, d_idx = form_trips(p)
        if len(raw_trips): raw_trips = ("VALUES ?e0 {%s}" %raw_trips[0], )

        if re.search('^Q', list(const_entities)[0]):
            const_type = 'mid'
            const_entiti = ['wd:%s' %e for e in const_entities]
            const = "Filter (?e0 != ?e%s).\nFilter (?e2 != ?e%s).\nVALUES ?e%s {%s}.\n?e%s rdfs:label ?name." %(t_idx+3, t_idx+3, t_idx+2, ' '.join(sorted(const_entiti)), t_idx+3)
            queries = [('?e%s ?r ?d%s' %(t_idx, t_idx+1), '?d%s ?r2 ?e%s' %(t_idx+1, t_idx+2),  \
                        '?d%s ?r3 ?e%s' % (t_idx + 1, t_idx + 3), 'MINUS {?d%s rdfs:label ?name.}' %(t_idx + 1))]
        elif re.search('^\d', list(const_entities)[0]):
            const_type = 'year'
            year = int(list(const_entities)[0])
            const = "FILTER(Year(?e%s) >= %s).\nVALUES ?r2 {pq:P582}.\n?e%s rdfs:label ?name3." % (t_idx + 2, year, t_idx + 3)
            queries = [('?e%s ?r ?d%s' %(t_idx, t_idx+1), '?d%s ?r2 ?e%s' %(t_idx+1, t_idx+2),  \
                        '?d%s ?r3 ?e%s' % (t_idx + 1, t_idx + 3), 'MINUS {?d%s rdfs:label ?name.}' %(t_idx + 1))]
            const1 = '.'.join(['ORDER BY (xsd:dateTime(?e%s))\nLIMIT 1' % (t_idx + 2)])
        elif list(const_entities)[0] in ['first', 'last', 'current', 'newly', 'final']:
            const_type = list(const_entities)[0]
            order = 'ASC' if const_type in ['first'] else 'DESC'
            const1 = '.'.join(['ORDER BY %s (xsd:dateTime(?e%s))\nLIMIT 1' % (order, t_idx + 2)])
            const = '.'.join(['?e%s rdfs:label ?name' % (t_idx + 1),
                              'FILTER (datatype(?e%s) in (xsd:dateTime))' % (t_idx + 2),
                              'VALUES ?r2 {wdt:P577}'])
            queries = [('?e%s ?r ?d%s' %(t_idx, t_idx+1), '?d%s ?r2 ?e%s' %(t_idx+1, t_idx+2),  \
                        '?d%s ?r3 ?e%s' % (t_idx + 1, t_idx + 3), 'MINUS {?d%s rdfs:label ?name.}' %(t_idx + 1))]
        else:
            return kbs, sparql_txts
            #raise Exception('SQL_2hop_reverse has wrong constraint format %s' %str(const_entities))

        for q_idx, query in enumerate(queries):
            trips = '.\n'.join(raw_trips + query)
            retu = '?e%s ?r3 ?e%s ?r2 ?r ?e%s' %(t_idx+3, t_idx+2, t_idx)
            sparql_txt = """SELECT DISTINCT %s WHERE {%s\n%s}%s""" %(retu, trips, const, const1)
            
            sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            try:
                if (QUERY is not None) and sparql_txt in QUERY: continue
                results = sparql.query().convert()
                if results['results']['bindings']:
                    for t in results['results']['bindings']:
                        
                        h = t['e%s' %(t_idx+3)]['value'].split('/entity/')[-1] if re.search('^http', t['e%s' %(t_idx+3)]['value']) else t['e%s' %(t_idx+4)]['value']
                        r3 = t['r3']['value'].split('/')[-1] if re.search('^http', t['r3']['value']) else t['r3']['value']
                        e = t['e%s' %(t_idx+2)]['value'].split('/entity/')[-1] if re.search('^http', t['e%s' %(t_idx+2)]['value']) else t['e%s' %(t_idx+2)]['value']
                        r2 = t['r2']['value'].split('/')[-1] if re.search('^http', t['r2']['value']) else t['r2']['value']
                        r = t['r']['value'].split('/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                        t = t['e%s' %(t_idx)]['value'].split('/entity/')[-1] if re.search('^http', t['e%s' %(t_idx)]['value']) else t['e%s' %(t_idx)]['value']
                        if const_type in ['mid']:
                            if q_idx == 0:
                                kbs[((t, r, '?d%s' %(t_idx+1)), ('?d%s' %(t_idx+1), r2, e), ('?d%s' %(t_idx+1), r3, '?e%s' %(t_idx+3)))].add(h)
                            else:
                                kbs[((h, r2, '?d%s' %(t_idx+1)), ('?d%s' %(t_idx+1), r, '?d%s' %d_idx))].add(t)
                        elif const_type in ['year']:
                            kbs[((t, r, '?d%s' % (t_idx + 1)), ('?d%s' % (t_idx + 1), r2, e), ('?d%s' % (t_idx + 1), r3, '?e%s' % (t_idx + 3)))].add(h)
                sparql_txts.add(sparql_txt)
            except urllib2.HTTPError as err:
                if err.code == 429:
                    time.sleep(60)
            except:
                pass

        return kbs, sparql_txts


    def wikidata_id_to_type(self, e):
        if (e in self.TYPE): return self.TYPE[e]
        
        name = u'O'
        sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
        sparql_txt = """SELECT DISTINCT ?t WHERE {wd:%s wdt:P31 ?t}LIMIT 1""" %(e)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            name = results['results']['bindings'][0]['t']['value'].split('/entity/')[-1] if results['results']['bindings'] else u'O'
        except urllib2.HTTPError as err:
            if err.code == 429:
                time.sleep(60); name = u'O'
        except:
            name = u'O'
        
        self.TYPE[e] = name

        return name


    def wikidata_id_to_out_degree(self, e):
        if (e in self.OUTDEGREE): return self.OUTDEGREE[e]
        
        name = 0
        sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
        sparql_txt = """SELECT (count(?a) as ?t) WHERE {wd:%s ?r ?a.}""" %(e)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            name = int(results['results']['bindings'][0]['t']['value']) if results['results']['bindings'] else 0
        except urllib2.HTTPError as err:
            if err.code == 429:
                time.sleep(60); name = 0
        except:
            name = 0

        self.OUTDEGREE[e] = name

        return name


    def wikidata_id_to_label(self, e):
        if not re.search('^[QP]', e):
            if is_year(e):
                print(e)
                return convert_year_to_timestamp(e)
            if is_date(e):
                return convert_date_to_timestamp(e)
            if re.search(' minute$', e):
                return re.sub(' minute$', '', e)
            return e
        
        if (e in self.M2N): return self.M2N[e]

        name = u'UNK'
        sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
        sparql_txt = """SELECT ?t WHERE {wd:%s rdfs:label ?t.
        FILTER (langMatches(lang(?t), 'en'))}""" %(e)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            name = results['results']['bindings'][0]['t']['value'] if results['results']['bindings'] else u'UNK'
        except urllib2.HTTPError as err:
            if err.code == 429:
                time.sleep(60); name = u'UNK'
        except:
            name = u'UNK'
        
        self.M2N[e] = name

        return name


    def wikidata_label_to_id(self, e, raw_const=None):
        '''
        :param e: name of topic entity, eg: 'Alcide De Gasperi'
        :param raw_const: tuple of previous topic entities for example, (id1, id2, ...)
        '''
        key = str((e, raw_const))
        if (key in self.M2N): return self.M2N[key]
        if (raw_const is not None):
            const = ['wd:%s' %c for c in raw_const]
            const = "?e1 ?r ?t. VALUES ?e1 {%s}" %(' '.join(const))
        else:
            const = ""

        name = u'UNK'
        sparql = SPARQLWrapper(self.SPARQLPATH, agent=AGENT)
        sparql_txt = """SELECT ?t WHERE {?t rdfs:label "%s"@en.%s} limit 10""" %(e, const)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            name = results['results']['bindings'][0]['t']['value'] if results['results']['bindings'] else u'UNK'
            name = name.split('/')[-1]
        except urllib2.HTTPError as err:
            if err.code == 429:
                time.sleep(60); name = u'UNK'
        except:
            name = u'UNK'

        self.M2N[key] = name;

        return name


# convert a date from the wikidata frontendstyle to timestamp style
def convert_date_to_timestamp (date):
	sdate = date.split(" ")
	# add the leading zero
	if (len(sdate[0]) < 2):
		sdate[0] = "0" + sdate[0]
	return sdate[2] + '-' + convert_month_to_number(sdate[1]) + '-' + sdate[0] + 'T00:00:00Z'


# convert a year to timestamp style
def convert_year_to_timestamp(year):
	return year + '-01-01T00:00:00Z'


# return if the given string is a literal or a date
def is_literal_or_date (answer):
	return not('www.wikidata.org' in answer)


# convert the given month to a number
def convert_month_to_number(month):
	return{
		"january" : "01",
		"february" : "02",
		"march" : "03",
		"april" : "04",
		"may" : "05",
		"june" : "06",
		"july" : "07",
		"august" : "08",
		"september" : "09",
		"october" : "10",
		"november" : "11",
		"december" : "12"
	}[month.lower()]


# return if the given string describes a year in the format YYYY
def is_year(year):
	pattern = re.compile('^[0-9][0-9][0-9][0-9]$')
	if not(pattern.match(year.strip())):
		return False
	else:
		return True


# return if the given string is a date
def is_date(date):
	pattern = re.compile('^[0-9]+ [A-z]+ [0-9][0-9][0-9][0-9]$')
	if not(pattern.match(date.strip())):
		return False
	else:
		return True


if __name__ == '__main__': 
    test()
    # my_sparql = SparqlRetriever()
    # cache_dir = "../CONVEX-cache/"
    # my_sparql.load_cache('%s/M2N.json' % cache_dir,
    #                      '%s/STATEMENTS.json' % cache_dir,
    #                      '%s/QUERY.json' % cache_dir,
    #                      '%s/TYPE.json' % cache_dir,
    #                      '%s/OUTDEGREE.json' % cache_dir)
    #print(my_sparql.SQL_1hop_reverse((('Q484523',),), ['last']))
    #print(my_sparql.SQL_2hop_reverse((('Q3127867',),), None))
    #print(my_sparql.wikidata_id_to_type('Q3918174'))
    #print(my_sparql.SQL_string_entities('Q494244'))
    #print(my_sparql.SQL_1hop((('Q494244',),)))
    #print(my_sparql.SQL_1hop_out((('Q6581097',),)))
    #print(my_sparql.SQL_1hop_interaction((('Q78885',), ('Q3707913',)), ['and']))
    #print(my_sparql.wikidata_label_to_id("dateModified"))
    #print(my_sparql.wikidata_id_to_label('P527'))

