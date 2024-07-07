from abc import ABC, abstractmethod
import requests
from typing import Any, Dict, List

import logging
import json
import os
import argparse
from pprint import pprint
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from bs4 import UnicodeDammit
from socket import timeout
import re
import unicodedata

with open('subscription_key.txt', 'r') as key:
    SUBSCRIPTION_KEY = key.read().rstrip()

CONTENT = 'content'
DEFAULT_NUM_TO_RETRIEVE = 5


class SearchEngineRetriever():
    """
    Queries a server (eg, search engine) for a set of documents.

    This module relies on a running HTTP server. For each retrieval it sends the query
    to this server and receives a JSON; it parses the JSON to create the response.
    """

    def __init__(self):
#        super().__init__(opt=opt)
        self.skip_query_token = None
        self.subscription_key = SUBSCRIPTION_KEY
        self.server_address = "https://api.bing.microsoft.com/v7.0/search"

    def create_content_dict(self, content: list, **kwargs) -> Dict:
        resp_content = {CONTENT: content}
        resp_content.update(**kwargs)
        return resp_content

    def _query_search_server(self, query_term, n):
        server = self.server_address
        params = {"q": query_term, "textDecorations": True, "textFormat": "HTML", "count": n}
        headers = { 'Ocp-Apim-Subscription-Key': self.subscription_key }
        logging.info(f'sending search request to {server}')
        server_response = requests.get(server, headers=headers, params=params)
        resp_status = server_response.status_code
        if resp_status == 200:
            result = server_response.json()
            try:
                return result["webPages"]["value"]
            except:
                pass
        logging.error(
            f'Failed to retrieve data from server! Search server returned status {resp_status}'
        )

    def _validate_server(self, address):
        if not address:
            raise ValueError('Must provide a valid server for search')
        if address.startswith('http://') or address.startswith('https://'):
            return address
        PROTOCOL = 'http://'
        logging.warning(f'No protocol provided, using "{PROTOCOL}"')
        return f'{PROTOCOL}{address}'

    def _retrieve_single(self, search_query: str, num_ret: int):
        if search_query == self.skip_query_token:
            return None

        retrieved_docs = []
        search_server_resp = self._query_search_server(search_query, 10) #num_ret
        if not search_server_resp:
            logging.warning(
                f'Server search did not produce any results for "{search_query}" query.'
                ' returning an empty set of results for this query.'
            )
            return retrieved_docs
        for rd in search_server_resp:
            url = rd.get('url', '')
            title = rd.get('name', '')
            content = self.get_details(url)
            snippet = rd.get("snippet", " ")
            if len(content) > 1 and num_ret > 0:
                retrieved_docs.append(
                    self.create_content_dict(url=url, title=title, content=[title]+content, snippet=snippet)
                )
                num_ret -= 1
            
        return retrieved_docs

    def postprocess(self, string):
        string = string.replace('<b>','')
        string = string.split('</b>')
        return string

    def get_details(self, url):
        abbreviations = {'dr.': 'doctor', 'mr.': 'mister', 'bro.': 'brother', 'bro': 'brother', 'mrs.': 'mistress', 'ms.': 'miss', 'jr.': 'junior', 'sr.': 'senior',
                 'i.e.': 'for example', 'e.g.': 'for example', 'vs.': 'versus'}
        terminators = ['.', '!', '?', ';']
        wrappers = ['"', "'", ')', ']', '}']

        def find_sentences(paragraph):
            end = True
            sentences = []
            while end > -1:
                end = find_sentence_end(paragraph)
                if end > -1:
                    sentences.append(paragraph[end:].strip())
                    paragraph = paragraph[:end]
            if len(paragraph) > 0:    
                sentences.append(paragraph)
            sentences.reverse()
            return sentences


        def find_sentence_end(paragraph):
            [possible_endings, contraction_locations] = [[], []]
            contractions = abbreviations.keys()
            sentence_terminators = terminators + [terminator + wrapper for wrapper in wrappers for terminator in terminators]
            for sentence_terminator in sentence_terminators:
                t_indices = list(find_all(paragraph, sentence_terminator))
                possible_endings.extend(([] if not len(t_indices) else [[i, len(sentence_terminator)] for i in t_indices]))
            for contraction in contractions:
                c_indices = list(find_all(paragraph, contraction))
                contraction_locations.extend(([] if not len(c_indices) else [i + len(contraction) for i in c_indices]))
            possible_endings = [pe for pe in possible_endings if pe[0] + pe[1] not in contraction_locations]
            if len(paragraph) in [pe[0] + pe[1] for pe in possible_endings]:
                max_end_start = max([pe[0] for pe in possible_endings])
                possible_endings = [pe for pe in possible_endings if pe[0] != max_end_start]
            possible_endings = [pe[0] + pe[1] for pe in possible_endings]
            # possible_endings = [pe[0] + pe[1] for pe in possible_endings if sum(pe) > len(paragraph) or (sum(pe) < len(paragraph) and paragraph[sum(pe)] == ' ')]
            end = (-1 if not len(possible_endings) else max(possible_endings))
            return end


        def find_all(a_str, sub):
            start = 0
            while True:
                start = a_str.find(sub, start)
                if start == -1:
                    return
                yield start
                start += len(sub)

        fixed_len = 100
        chunk_num = 3 # 5
        final_text = []
        string_len = 0
        update_len = 0
        headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                }
        try:
            try:
                r = requests.get(url, headers=headers, timeout=5)
                html = r.text.encode('iso-8859-1').decode('gbk')
            except:
                r = requests.get(url, headers=headers, timeout=5)
                html = r.text
            htmlParse = BeautifulSoup(html, 'html.parser')
            
        except:
            return []
        
        raw_para = ""
        for para in htmlParse.find_all("p"):
            sents = " ".join(para.get_text().strip().split())
            sents = unicodedata.normalize("NFKD",sents)
            sents = re.sub(r'\n', '', sents)
            sents = re.sub(r'\t', '', sents)
            if len(sents) > 0:
                raw_para += ' ' + sents

        sent_list = find_sentences(raw_para)
        
        final_text = sent_list
        return final_text

    def retrieve(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        # TODO: update the server (and then this) for batch responses.
        return [self._retrieve_single(q, num_ret) for q in queries]

