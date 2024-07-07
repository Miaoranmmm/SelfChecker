from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.agents import Tool, initialize_agent, tool, AgentExecutor, ZeroShotAgent
import os
import re
import jsonlines
import json
import argparse
import sqlite3
from src.search_engine import *
from utils import *
from dotenv import load_dotenv
load_dotenv()
os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://llmaugmenter.openai.azure.com/" # "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

TEMPERATURE = 0.2
DEPLOYMENT_NAME = "text003"
MODEL_NAME = "text003"
MAX_LENGTH = 4096
SearchEngine = SearchEngineRetriever()
llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)

prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt/prompt_examples')

def search_engine(query):
    search_result = SearchEngine.retrieve([query], 3)[0]
    search_result_str = '\n'.join(record['snippet'] for record in search_result)
    return search_result_str

db = 'wiki_pages.db'
conn = sqlite3.connect(db)
c = conn.cursor()
def normalize_title(name):
    name = name.strip()
    name = name.replace('-LRB-', '').replace('-RRB-', '').replace('-LSB-', '').replace('-RSB-','')
    name = name.replace('-COLON-', '')
    name = name.replace('_', ' ')
    return name

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

def normalize_bracket(string):
    string = string.strip()
    string = string.replace('-LRB-','(')
    string = string.replace('-RRB-',')')
    string = string.replace('-LSB-','[')
    string = string.replace('-RSB-',']')
    return string

def querydb(name, like_search=False, similar=False):
    name = normalize_name(name)
    name = normalize_title(name)
    lines_query = f"SELECT * from wikipages WHERE norm_id='{name}'"
    c.execute(lines_query)
    lines = c.fetchone()
    if not lines and like_search:
        lines_query = f"SELECT * from wikipages WHERE norm_id LIKE '%{name}%' LIMIT 10"
        c.execute(lines_query)
        lines = c.fetchone()
    if similar:
        lines_query = f"SELECT * from wikipages WHERE norm_id LIKE '%{name.replace(' ', '%')}%' LIMIT 10"
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
      

def search_db(query, similar=False, similar_queries=True, like_search=True):
    search_result = {}
    candidate_passage_names = []
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
            if not isinstance(lines, list):
                lines = [lines]
            for line in lines:
                title, wiki_passage, p = postprocess_results(line)
                candidate_passage_names.append(title)
                search_result[title] = wiki_passage
                return title + '\n' + wiki_passage

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--save_file",
        type=str,
        default='results_react.json',
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
        "--example_id",
        type=int,
        default=None,
        help="example index",
    ) 
    parser.add_argument(
        "--test_file",
        type=str,
        default='../fever/paper_test.jsonl',
        help="test file path",
    ) 
    parser.add_argument(
        "--example_file",
        type=str,
        default='prompts/prompt_exmples/fever/cot_prompt.jsonl',
        help="test file path",
    )
    parser.add_argument(
        "--verdict_labels",
        type=str,
        default='SUPPORT, REFUTE, NOT ENOUGH INFO',
        help="test file path",
    )  
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default='fever / wice / bingcheck',
        help="test file path",
    )
    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    @tool("Search", return_direct=False)
    def search(input_text: str) -> str:
        """useful for when you need to ask with search"""
        # return search_engine(input_text)
        if args.eval_dataset == 'fever':
            return search_db(input_text)
        else:
            return search_engine(input_text)

    
    tools = [search]
    tool_introduction = '\n'.join([f"{tool.name}: {tool.description}" for tool in tools])
    actions = ', '.join([tool.name for tool in tools])
    tool_names = [tool.name for tool in tools]


    EXAMPLE_LISTS = f""""Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action: Search
Action Input: Nikolaj Coster-Waldau
Observation: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series..
Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought: Because he "appeared in the 2009 Fox television film Virtuality", he should have worked with the Fox Broadcasting Company.
Final Answer: SUPPORTS
"""

    template = f"""Imagine you are an expert in determining the factualness of claims. Please determine the verdict label for the given claim. The candidate labels can be {args.verdict_labels}.
You have access to the following tools:

{tool_introduction}

Use the following format:

Input: the response of language model to the user query. you must verify the factual accuracy of the response.
Thought: you should always realize what you have known and think about what to do and which tool to use. 
Action: the action to take, it must be {actions} or directly give the final answer
Action Input: the input to the action, must follow instructions of tools
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I can give an answer based on the evidence
Final Answer: a summary of verdict result of subclaims, such as the number of subclaims and the number of subclaims in each label class
"""
    template += "\n\nBelow is an example:\n\n"
    template += EXAMPLE_LISTS
    template += "\nBegin!"
    template += "\n\nInput: {claim}\n{agent_scratchpad}"
    

    complete_prompt = PromptTemplate(
        template=template,
        input_variables=["claim","agent_scratchpad"]
    )

    
    llm_chain = LLMChain(llm=llm, prompt=complete_prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, prompt=complete_prompt)
    agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    

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
                    result_file.append(obj)
            output_file = jsonlines.open(os.path.join(args.save_dir, args.save_file), 'a')
        # print(len(result_file))
    elif file_name.endswith('jsonl'):
        output_file = jsonlines.open(os.path.join(args.save_dir, args.save_file), 'w')
        result_file = []
    elif file_name.endswith('json'):
        result_file = {}

    if args.eval_dataset == 'fever':
        with open('../fever/sampled_ids.json') as sample_file:
            sampled_ids = json.load(sample_file)
        with open(args.test_file, "r", encoding="utf-8") as reader:
            idx = 0
            for item in jsonlines.Reader(reader):
                if int(item['id']) not in sampled_ids:
                    continue
                if str(item['id']) in result_file:
                    continue
                if args.start_id or args.end_id:
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
                if args.example_id:
                    if item['id'] != args.example_id:
                        continue
                
                print(item['id'])
                try:
                    response = agent({"claim":item['claim']})['output']
                    print(response) 
                except:
                    idx += 1
                    continue
                predicted_label = response
                print(predicted_label)
                print(item['label'])
                record = {
                    'claim': item['claim'],
                    'conclusion': predicted_label,
                    'label': item['label'],
                }
                if args.save_file:
                    write_json(item['id'], record, os.path.join(args.save_dir, args.save_file))
                        
                print(idx)
                idx += 1
                print('='*60)
    
    elif args.eval_dataset == 'bingcheck':
        with open(args.test_file, "r", encoding="utf-8") as reader:
            idx = 0 + len(result_file)
            print(len(result_file))
            data = json.load(reader)
            for item_id in data:
                item = data[item_id]
                if args.start_id or args.end_id:
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
                if args.example_id:
                    if item_id != args.example_id:
                        continue
                if item_id in result_file:
                    idx += 1
                    continue

                
                input_response = re.sub('\[\^.*?\^\]', '', item['response'].replace('\n',' ').replace('  ',' ').replace('**',''))
                response = agent({'claim':input_response})
                # try:
                #     response = agent({'claim':input_response})
                # except:
                #     continue
                
                record = {'claim':input_response, 'conclusion':response['output']}
                for step in response['intermediate_steps']:
                    action, inter_result = step
                    if action.tool == 'Search':
                        record['search_query'] = action.tool_input
                        record['search_result'] = inter_result
                
                if args.save_file:
                    write_json(item['id'], record, os.path.join(args.save_dir, args.save_file))
                        
                print(idx)
                idx += 1
                print('='*60)

    
if __name__=='__main__':
    main()