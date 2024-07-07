from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
import tiktoken
import os
import re
import jsonlines
import json
import argparse
import sqlite3
import numpy as np
import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from retry import retry
from prompts.prompt_fever import *
from collections import Counter
from utils import *
from dotenv import load_dotenv
load_dotenv()
os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://llmaugmenter.openai.azure.com/" # "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview" #"2023-05-15"

regex = r'\<(.*)\>'
punctuation = r'[,.;:@#?!&$]+\ *'
stop_words = set(stopwords.words('english'))

MAX_LENGTH = 4096
TEMPERATURE = 0.2
DEPLOYMENT_NAME = "text003"
MODEL_NAME = "text003"

query_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)
evidence_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)
result_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)
llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)
encoding = tiktoken.get_encoding("p50k_base")

search_res = {}
search_queries = ''
verification_evidence = ''
claim_to_verify = ''


def normalize_title(name):
    name = name.strip()
    name = name.replace('-LRB-', '').replace('-RRB-', '').replace('-LSB-', '').replace('-RSB-','')
    name = name.replace('-COLON-', '')
    name = name.replace('_', ' ')
    return name

def normalize_bracket(string):
    string = string.strip()
    string = string.replace('-LRB-','(')
    string = string.replace('-RRB-',')')
    string = string.replace('-LSB-','[')
    string = string.replace('-RSB-',']')
    return string

def replace_bracket(string):
    string = string.strip()
    string = string.replace('(','-LRB-')
    string = string.replace(')','-RRB-')
    string = string.replace('[','-LSB-')
    string = string.replace(']','-RSB-')
    return string

## Query Generation ##
def generate_query(claim, temp=None):
    global search_queries
    if temp:
        query_llm_new = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=temp)
        search_queries = query_llm_new(query_fs_prompt.format(claim=claim))
        search_queries = search_queries.split('\n')[0].replace('<|im_end|>','').strip()
    search_queries = query_llm(query_fs_prompt.format(claim=claim))#.split('\n')[0].replace('<|im_end|>','').strip()
    return search_queries

## Database ##
db = 'wiki_pages.db'
conn = sqlite3.connect(db)
c = conn.cursor()

def normalize_name(name):
    name = name.strip()
    # name = name.replace(' ','_')
    name = name.replace('"','')
    name = name.replace("'","''")
    name = name.replace('(','-LRB-')
    name = name.replace(')','-RRB-')
    name = name.replace('[','-LSB-')
    name = name.replace(']','-RSB-')
    return name

def querydb(name, like_search=False, similar=False):
    name = normalize_name(name)
    name = normalize_title(name)
    lines_query = f"SELECT * from wikipages WHERE norm_id='{name}'"
    c.execute(lines_query)
    lines = c.fetchone()
    if not lines and like_search:
        lines_query = f"SELECT * from wikipages WHERE norm_id LIKE '%{name}%' LIMIT 20"
        c.execute(lines_query)
        lines = c.fetchone()
    if similar:
        lines_query = f"SELECT * from wikipages WHERE norm_id LIKE '%{name.replace(' ', '%')}%' LIMIT 20"
        c.execute(lines_query)
        lines = [lines] if lines else []
        lines += c.fetchall()
        lines = list(set(lines))

    return lines


def postprocess_results(lines):
    title = lines[0]
    lines = lines[-1]
    lines = lines.split('\n')
    lines = [normalize_bracket(' '.join(sent.split('\t')[1:])) for sent in lines]
    wiki_passage = '\n'.join([f"{idx}\t{sent}" for idx, sent in enumerate(lines)])
    return title, wiki_passage, lines
      

def search(query, similar=False, similar_queries=True, like_search=True):
    print('Start searching')
    global search_res
    earch_result = {}
    candidate_passage_names = []
    passage_names = []
    query_norm = normalize_bracket(query.replace('_',' '))
    query_list = query_norm.split(' ; ')
    result = 'no result found in the database'
    if similar_queries:
        queries = set()
        for query in query_list:
            queries.add(query.strip())
            queries.add(re.sub('[^a-zA-Z0-9 \n\.]', '', query).strip().replace('  ',' '))
            queries.add(re.sub(r"\((.*?)\)", "", query).strip())
        queries = list(queries)
    else:
        queries = query_list
    for query in queries:
        q = query.replace('  ', ' ')
        lines = querydb(q, like_search=True, similar=similar)
        
        if lines:
            print(f'got search results for {q}')
            if not isinstance(lines, list):
                lines = [lines]
            for line in lines:
                title, wiki_passage, p = postprocess_results(line)
                candidate_passage_names.append(title)
                search_result[title] = wiki_passage
        else: # no search result
            q_tokens = word_tokenize(q)
            filtered_q = ' '.join([w for w in q_tokens if w.lower() not in stop_words])
            if filtered_q != q:
                lines = querydb(filtered_q, like_search=True, similar=similar)
                # lines = search_wiki(filtered_q, like_search=True, similar=similar)

                if lines:
                    print(f'got search results for {filtered_q}')
                    if not isinstance(lines, list):
                        lines = [lines]
                    for line in lines:
                        title, wiki_passage, p = postprocess_results(line)
                        candidate_passage_names.append(title)
                        search_result[title] = wiki_passage
    print("Searching completed")
    if len(candidate_passage_names) > 0:
        if len(candidate_passage_names) > 3:
            feedback = f"{' ; '.join(queries)}\nNow the retrieved passage titles are: {' ; '.join(candidate_passage_names)} .Please choose up to three relevant titles from the retrieved titles. Please copy the titles and do not alter the titles, even their letter casing or punctuations.\nSelected Titles:"
            result = query_llm(query_fs_prompt.format(claim=claim_to_verify) + feedback).replace('<|im_end|>','')#.split('\n')[0].strip()
            result = re.split(';|\n|,', result)
            passage_names = [passage_name.strip().replace(' ','_') for passage_name in result if passage_name.strip().replace(' ','_') in candidate_passage_names]
            if len(passage_names) < 1 or result == '':
                passage_names = random.choices(candidate_passage_names, weights = [1]*len(candidate_passage_names), k=3)
                
        else:
            passage_names = candidate_passage_names
        
        
        for p_name in passage_names:
            p_name = p_name.strip()
            try:
                search_res[p_name] = search_result[p_name]
            except:
                pass
    
    return ' ; '.join(passage_names)

## Evidence Selection ##
def select_evidence(claim_passage, similar_queries=True):
    print("Start evidence selection")
    global search_res
    global verification_evidence
    claim = claim_passage.split('|')[0].strip()
    names = claim_passage.split('|')[1].strip()
    
    passages = []
    evidence_list= []
    candidate_evidence = {}
    selected_evidences = []
    score = 0
    conclusion = 'not enough info'
    evidence = 'no evidence'
    analysis = []

    for title in search_res:
        wiki_passage = search_res[title].split('\n') 
        n_lines = len(wiki_passage)
        for i in range(0, n_lines, 5):
            passage = '\n'.join(wiki_passage[i: min(n_lines, i+5)])
            evidence_fs_prompt_truncated = evidence_fs_prompt
            selection_input = evidence_fs_prompt.format(claim=claim, passage=title + '\n' + passage)
            example_num = 14
            while (len(encoding.encode(selection_input)) > (MAX_LENGTH - 400)) and example_num > 0: 
                evidence_fs_prompt_truncated = FewShotPromptTemplate(
                    examples=evidence_example[:example_num],
                    example_prompt=evidence_prompt,
                    prefix=evidence_selection_prefix,
                    suffix="\nNow, let's focus on your task. You are given a claim and a passage. Please read the passage carefully and find lines that contain information supporting or refuting the claim.\n\nClaim: {claim}\nPassage:\n{passage}\nLet's think step by step.\nAnalysis:",
                    input_variables=["claim", "passage"],
                    example_separator="\n",
                )
                selection_input = evidence_fs_prompt_truncated.format(claim=claim, passage=title + '\n' + passage)
                example_num -= 1
                
            try:
                cur_analysis = evidence_llm(selection_input).strip()
                analysis.append(cur_analysis.replace('\n', ';').replace(' lines ',' line '))
            except:
                continue

            try:    
                s_index = re.sub(punctuation, ' ', cur_analysis.split(' line ')[1]).split(' because ')[0].replace('-',' - ').replace('  ',' ').split()
                if '-' in s_index:
                    replace_idx = s_index.index('-')
                    replace_range = list(range(int(s_index[replace_idx-1]), int(s_index[replace_idx+1])+1))
                    replace_range = [str(idx) for idx in replace_range]
                    before = s_index[:replace_idx-1] if replace_idx > 1 else []
                    after = s_index[replace_idx+2:] if replace_idx+2 < len(s_index) else []
                    s_index = before + replace_range + after
                for s_idx in s_index:
                    s_idx = s_idx.strip()
                    sent = '\t'.join(wiki_passage[int(s_idx)].strip().split('\t')[1:])
                    if len(sent) > 0:
                        selected_evidences.append(sent)
                        if title in candidate_evidence:
                            candidate_evidence[title].append(f"{s_idx}\t{sent}")
                        else:
                            candidate_evidence[title] = [f"{s_idx}\t{sent}"]

                        # evidence_list.append(f'<{replace_bracket(title)}> line {s_idx}: {sent}')
            except:
                pass
    print('Selection Finished')
    candidate_evidence_str = ''
    if len(candidate_evidence) > 0:
        for candidate_title in candidate_evidence:
            candidate_evidence_str += candidate_title
            candidate_evidence_str += '\n'
            candidate_evidence_str += '\n'.join(candidate_evidence[candidate_title])
            candidate_evidence_str += '\n'

        if len(encoding.encode(entailment_fs_prompt.format(claim=claim, evidence=candidate_evidence_str))) < MAX_LENGTH - 256:
            evidence_list = [f"<{p_title}> {piece}" for p_title in candidate_evidence for piece in candidate_evidence[p_title]]
            return ' | '.join(evidence_list)

        final_selection_input = evidence_fs_prompt.format(claim=claim, passage=candidate_evidence_str)
        example_num = 15
        while len(encoding.encode(final_selection_input)) > MAX_LENGTH - 256 and example_num > 0: 
            evidence_fs_prompt_truncated = FewShotPromptTemplate(
                examples=evidence_example[:example_num],
                example_prompt=evidence_prompt,
                prefix=evidence_selection_prefix,
                suffix="\nNow, let's focus on your task. You are given a claim and a passage. Please read the passage carefully and find lines that contain information supporting or refuting the claim.\n\nClaim: {claim}\nPassage:\n{passage}\nLet's think step by step.\nAnalysis:",
                input_variables=["claim", "passage"],
                example_separator="\n",
            )
            final_selection_input = evidence_fs_prompt_truncated.format(claim=claim, passage=candidate_evidence_str)
            example_num -= 1
        
        final_analysis = evidence_llm(final_selection_input).strip().replace(' lines ',' line ')
        
        try:    
            s_index = re.sub(punctuation, ' ', final_analysis.split(' line ')[1]).split(' because ')[0].replace('-',' - ').replace('  ',' ').split()
            if '-' in s_index:
                replace_idx = s_index.index('-')
                replace_range = list(range(int(s_index[replace_idx-1]), int(s_index[replace_idx+1])+1))
                replace_range = [str(idx) for idx in replace_range]
                before = s_index[:replace_idx-1] if replace_idx > 1 else []
                after = s_index[replace_idx+2:] if replace_idx+2 < len(s_index) else []
                s_index = before + replace_range + after
                
            for s_idx in s_index:
                s_idx = s_idx.strip()
                sent = '\t'.join(wiki_passage[int(s_idx)].strip().split('\t')[1:])
                if len(sent) > 0:
                    # print(f'SENT: {sent}')
                    evidence_list.append(f'<{replace_bracket(title)}> line {s_idx}: {sent}')                  
        except:
            for candidate_title in candidate_evidence:
                evd_pieces = candidate_evidence[candidate_title]
                for evd_piece in evd_pieces:
                    s_idx = evd_piece.split('\t')[0]
                    sent = '\t'.join(evd_piece.split('\t')[1:])
                    evidence_list.append(f'<{replace_bracket(candidate_title)}> line {s_idx}: {sent}')

        if len(evidence_list) > 0:
            evidence = ' | '.join(evidence_list)
        
    return evidence #, conclusion


## Verdict Prediction
def verdict_prediction(claim_evidence):
    global search_res
    try:
        claim = claim_evidence.split(' | ')[0]
        evidence_indices = claim_evidence.split(' | ')[1].split(' ; ')
        if not isinstance(evidence_indices, list):
            evidence_indices = [evidence_indices]
    except:
        return "The input format is incorrect. the input must be in the format: claim to verify | <title_of_evidence> line line_index_of_evidence. Please double-check and try again."

    evidence_list = []
    for evidence_index in evidence_indices:
        try:
            title, line_id = evidence_index.split(' line ')
            line_id = line_id.split(' ')[0]
            title = title.replace('<','').replace('>','')
            if title in search_res:
                passage = search_res[title].split('\n')
                evd_sentence = '\t'.join(passage[int(line_id)].split('\t')[1:])
                evidence_list.append(evd_sentence)
            else:
                lines = querydb(title)
                
                if lines:
                    title, wiki_passage, p = postprocess_results(lines)
                    evd_sentence = '\t'.join(wiki_passage[int(line_id)].split('\t')[1:])
                    evidence_list.append(evd_sentence)
        except:
            continue
        
    analysis = result_llm(entailment_fs_prompt.format(claim=claim, evidence='\n'.join(evidence_list))).replace('_', ' ').split('\n\n')[0].strip()
    final_result_prompt = f"{analysis}\nBased on the conclusion, predict the final verdict label of the claim. The valid labels include 'supports', 'refutes', and 'not enough info'.\nLabel:"
    result = result_llm(entailment_fs_prompt.format(claim=claim, evidence='\n'.join(evidence_list)) + '\n' + final_result_prompt).replace('_', ' ').lower()
    if 'refute' in result:
        result = 'refutes'
    elif 'support' in result and 'not support' not in result:
        result = 'supports'
    else:
        result = 'not enough info'
    return analysis + '\n' + f"The result is {result}"

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--save_file",
        type=str,
        default='results.json',
        help="path of output file",
    ) 
    parser.add_argument(
        "--save_dir",
        type=str,
        default='results/fever',
        help="dir of output file",
    ) 
    parser.add_argument(
        "--start_id",
        type=int,
        default=None,
        help="start index",
    )    
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="end index",
    ) 
    parser.add_argument(
        "--start_example_id",
        type=int,
        default=None,
        help="start example index",
    ) 
    parser.add_argument(
        "--eval_mode",
        type=str,
        default='full_pipeline',
        help="'full_pipeline', 'gold_passage', 'gold_evidence'",
    )
    parser.add_argument(
        "--similar",
        action='store_true',
        help="similar retrieval results",
    )
    parser.add_argument(
        "--example_id",
        type=int,
        default=None,
        help="example index",
    ) 
    args = parser.parse_args()

    return args

def main():
    global search_res
    global search_queries
    global claim_to_verify
    global evidence_llm
    global result_llm

    args = parse_args()

    random.seed(2023)

    @tool("QueryWriter", return_direct=False)
    def query_writer(claim: str) -> str:
        """useful for generating the search query to retrieve information. the input should be the claim to verify"""
        return generate_query(claim)

    @tool("Database", return_direct=False)
    def database(query: str) -> str:
        """useful when you have known candidate search queries and need to search wikipedia pages from the database. the input should be the results of query writer"""
        if args.similar:
            return search(query, similar=True)
        else:
            return search(query)

    @tool("EvidenceSeeker", return_direct=False)
    def evidence_seeker(claim_passage: str) -> str:
        """
        useful for reading passages and finding evidence. the input must be in the format: claim to verify | passage names from the database
        """
        return select_evidence(claim_passage)

    @tool("VerdictPredictor", return_direct=False)
    def verdict_predictor(claim_evidence: str) -> str:
        """
        useful for predicting the verdict label of the claim after obtaing evidence. the input should include the claim to verify and the index of retrieved evidence. the input must be in the format: claim to verify | <title_of_evidence> line line_index_of_evidence. if there are multiple evidence pieces; use ' ; ' as separator.
        """
        return verdict_prediction(claim_evidence)
        # return select_evidence(claim_passage)
    

    if args.eval_mode == 'full_pipeline':
        tools = [
            query_writer,
            database,
            evidence_seeker,
            verdict_predictor
        ]
        thought = (
        '1. first generate candidate query for retrieval.'
        '2. after having candidate search queries, use them to search for passages from the database.'
        '3. after getting passage names from the database, try to find evidence lines that supports / refutes the claim from the passages. Make sure the input format is correct.'
        '4. after retrieving evidence, determine the verdict label of the claim'
        'you can go back to previous steps if you need to find more information'
        )
        template_examples = EXAMPLES_FULL_PIPELINE
        template_prefix = "\n\nClaim: {claim}\n{agent_scratchpad}"
        template_variables = ["claim","agent_scratchpad"]

    elif args.eval_mode == 'gold_passage':
        tools = [
            evidence_seeker,
            verdict_predictor
        ]
        thought = (
        '1. given the passage names and the claim to verify, try to find evidence lines that supports / refutes the claim from the passages. Make sure the input format is correct.'
        '2. determine the verdict of the claim based on the evidence'
        'you can go back to previous steps if you need to find more information'
        )
        template_examples = EXAMPLES_GOLD_PASSAGE
        template_prefix = "\n\nClaim: {claim}\nPassage Names:{query}\n{agent_scratchpad}"
        template_variables = ["claim", "query", "agent_scratchpad"]

    elif args.eval_mode == 'gold_evidence':
        tools = [
            verdict_predictor
        ]
        thought = (
        '1. determine the verdict of the claim based on the evidence'
        'you can go back to previous steps if you need to find more information'
        )
        template_examples = EXAMPLES_GOLD_EVIDENCE
        template_prefix =  "\n\nClaim: {claim}\nEvidence: {evidence}\n{agent_scratchpad}"
        template_variables = ["claim", "evidence", "agent_scratchpad"]

    else:
        raise Exception("Unexpected evaluation mode")

    tool_introduction = '\n'.join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = [tool.name for tool in tools]
    actions = ', '.join([tool.name for tool in tools])
    
    template = f"""Try your best to determine if the given claim is supported or refuted by Wikipedia pages or there is not enough infomation. You have access to the following tools:

{tool_introduction}

Use the following format:

Claim: the claim you must verify
Thought: you should always realize what you have known and think about what to do and which tool to use. Normally you should follow the steps: {thought}
Action: the action to take, should be one of [{actions}]
Action Input: the input to the action, must follow instructions of tools
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I can give an answer based on the evidence
Final Answer: should be in the form: the claim is suppoted / refuted or there is not enough information

"""
#Learn from the examples below:
   
    template += template_examples
    template += "\nNow please focus on your task."
    template += template_prefix
    
    complete_prompt = PromptTemplate(
        template=template,
        input_variables=template_variables
    )

    llm_chain = LLMChain(llm=llm, prompt=complete_prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, prompt=complete_prompt)
    agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    file_name = args.save_file
    if os.path.exists(os.path.join(args.save_dir,args.save_file)):
        if file_name.endswith('json'):
            with open(os.path.join(args.save_dir,args.save_file)) as existing:
                result_file = {}
                result_file = json.load(existing)
        else:
            with jsonlines.open(os.path.join(args.save_dir, args.save_file)) as reader:
                result_file = []
                for obj in reader:
                    result_file.append(obj['id'])
    else:
        result_file = {}

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fever/sampled_ids.json')) as sample_file:
        sampled_ids = json.load(sample_file)
        print(f'# sampled_ids: {len(sampled_ids)}')

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fever/paper_test.json'), "r", encoding="utf-8") as reader:
        idx = 0
        for item in jsonlines.Reader(reader):
            search_res = {}
            search_queries = ''
  
            if int(item['id']) not in sampled_ids:
                continue
            
            if args.start_id or args.end_id:
                # print(item['id'])
                if args.start_id and args.end_id:
                    if idx < args.start_id:
                        idx += 1
                        continue
                    elif idx >= args.end_id:
                        break
                elif args.start_id and idx < args.start_id:
                    idx += 1
                    continue
                elif args.end_id and idx not in range(args.end_id):
                    break
                elif item['id'] in result_file:
                    idx += 1
                    continue

            if args.example_id:
                if item['id'] != args.example_id:
                    continue

            print(item['id'])
            claim_to_verify = item['claim']

            if args.eval_mode == 'gold_passage':
                if item['label'] == 'NOT ENOUGH INFO':
                    temp = 0
                    while len(search_queries) < 1:
                        query = generate_query(item['claim'], temp=temp) # retrieve related passages
                        if query != '':
                            passage_name = search(query, similar=True).split(' ; ')[0]
                            lines = querydb(passage_name, like_search=True)
                            
                            if lines:
                                title, wiki_passage, p = postprocess_results(lines)
                                search_res[title] = wiki_passage
                                search_queries = normalize_title(title)
                        temp += 0.1
                        temp = min(temp, 2)
                else: 
                    search_queries = ' ; '.join(list(set([normalize_title(piece[0][2]) for piece in item['evidence']])))
                    search_result = search(search_queries, similar_queries=False)
                response = agent({"claim":item['claim'], "query": search_queries})
                # try:
                #     print('Passage name:',search_queries)
                #     print('Need to search')
                #     search_result = search(search_queries, similar_queries=False)
                #     response = agent({"claim":item['claim'], "query": search_queries})
                # except:
                #     idx += 1
                #     continue
                    
            elif args.eval_mode == 'gold_evidence':
                if item['label'] == 'NOT ENOUGH INFO':
                    temp = 0
                    evidence = None
                    while not evidence:
                        query = generate_query(item['claim'], temp=temp) # retrieve related passages
                        print(query)
                        if query != '':
                            passage_name = search(query, similar=True).split(' ; ')[0]
                            print('Passage name:',passage_name)
                            lines = querydb(passage_name, like_search=True)
                            # lines = search_wiki(passage_name)

                            if lines:
                                title, wiki_passage, p = postprocess_results(lines)
                                search_queries = normalize_title(title)
                                search_res[title] = wiki_passage
                                evi_idx = random.choices(range(len(p)),k=2)
                                evidence = [f"<{title}> line {line_idx}" for line_idx in evi_idx]
                                print('EVIDENCE:', evidence)
                        temp += 0.1
                        temp = min(temp, 2)
                else: 
                    evidence_name = [piece[0][-2] for piece in item['evidence']]
                    evidence_idx = [piece[0][-1] for piece in item['evidence']]
                    evidence = set()
                    for title, line_idx in zip(evidence_name, evidence_idx):
                        evidence.add(f"<{title}> line {line_idx}")
                    search_queries  = ' ; '.join(list(set(evidence_name)))
                    evidence = list(evidence)
                    print('EVIDENCE:', evidence)
                    evidence = ' ; '.join(evidence)
                    search_result = search(search_queries, similar_queries=False)
                   
                try:  
                    response = agent({"claim":item['claim'], "evidence": evidence})
                except:
                    idx += 1
                    continue

            else:
                response = agent({"claim":item['claim']})
                print(response['output'])
                idx += 1
                
  
            print(item['label'])
            record = {
                'claim': item['claim'],
                'conclusion': response['output'],
                'label': item['label'],
                'gold_evi': ' ; '.join([f"<{piece[2]}> line {piece[-1]}" for piece in item['evidence'][0]])
            }
            for step in response['intermediate_steps']:
                action, inter_result = step
                if action.tool == 'QueryWriter':
                    record['query'] = inter_result.strip()
                if action.tool == 'EvidenceSeeker':
                    record['evidence'] = inter_result.split('\n')[0]
                if action.tool == 'VerdictPredictor':
                    record['verdict_analysis'] = inter_result
            if item['id'] in result_file:
                del result_file[item['id']]
            result_file[item['id']] = record
            
            if args.save_file:
                write_json(item['id'], record, os.path.join(args.save_dir, args.save_file))
            
            print(idx)
            idx += 1
            print('='*60)

if __name__=='__main__':
    main()