# from langchain import OpenAI, SQLDatabase, SQLDatabaseChain, LLMChain, PromptTemplate, FewShotPromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
import tiktoken
import os
import jsonlines
import json
import argparse
from collections import Counter
from prompts.prompt_bingcheck import *
from src.search_engine import *
from dotenv import load_dotenv
load_dotenv()
os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://llmaugmenter.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

SearchEngine = SearchEngineRetriever()
MAX_LENGTH = 4096
TEMPERATURE = 0
DEPLOYMENT_NAME = "text003"
MODEL_NAME = "text003"

input_str = ''
search_res = {}
search_queries = []
claim_splited = []
final_evidence = {}
final_conclusion = {}

split_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE) 
query_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE) 
evidence_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE) 
result_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)
llm_policy = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)
encoding = tiktoken.get_encoding("p50k_base")

def claim_split(claim):
    global input_str
    global claim_splited
    if claim != input_str:
        claim = input_str
    result = split_llm(split_fs_prompt.format(input=claim)).split('\n\n')[0].strip()
    claim_splited = result.split('\n')
    return result

def generate_query(claims):
    global search_queries
    global claim_splited
    claim_list = claims.split('[EOS]')
    if len(claim_list) != len(claim_splited):
        claim_list = claim_splited

    for claim in claim_list:
        query = ''
        feedback = ''
        while len(query) < 1:
            query = query_llm(query_fs_prompt.format(context='\n'.join(claim_list),claim=claim) + feedback).split('\n')[0].replace('_','').strip()
            feedback = '\nThe search query is not appropriate. Please consider the claim carefully and try again.\n\nQuery:'
        search_queries.append(query)
    return '[EOS]'.join(search_queries)

def search(query):
    search_result = SearchEngine.retrieve([query], 3)[0]
    return search_result

def process_passage(passage):
    lines = []
    for idx, line in enumerate(passage):
        lines.append(line.strip())
    return lines

def select_evidence(claim_query):
    global claim_splited
    global search_queries
    claims = []
    queries = []
    try:
        claims = claim_query.split(' | ')[0].split('[EOS]')
        queries = claim_query.split(' | ')[1].split('[EOS]')
    except:
        pass
    if claims != claim_splited:
        claims = claim_splited
    if queries != search_queries:
        queries = search_queries
    
    global search_res
    global final_evidence

    evidence_conclusion = ''

    for claim_idx, claim in enumerate(claims):
        if claim in search_res:
            records = search_res[claim]
            passages = [record['content'] for record in records]
            records = search_res[claim]
        else:
            records = search(queries[claim_idx])
            passages = []
            search_res_list = []
            for record in records:
                result = {
                    'title': record['title'],
                    'url': record['url'],
                    'content': process_passage(record['content'])
                }
                passages.append(process_passage(record['content']))
                search_res_list.append(result)
            search_res[claim] = search_res_list

        claim_evidence = []

        for passage in passages:
            n_lines = len(passage)
            for i in range(0, n_lines, 10):
                p_str = '\n'.join(passage[i: min(n_lines, i+10)])
                evidence_fs_prompt_truncated = evidence_fs_prompt
                selection_input = evidence_fs_prompt.format(claim=claim, passage=p_str)
                example_num = 15
                while len(encoding.encode(selection_input)) > MAX_LENGTH - 256 and example_num > 0: 
                    evidence_fs_prompt_truncated = FewShotPromptTemplate(
                        examples=evidence_example[:example_num],
                        example_prompt=evidence_prompt,
                        prefix=evidence_selection_prefix,
                        suffix="\nNow, let's focus on your task. You are given a claim and a passage. Please read the passage carefully and copy sentences that contain information supporting or refuting the claim.\n\nClaim: {claim}\nPassage:\n{passage}\nAnalysis:",
                        input_variables=["claim", "passage"],
                        example_separator="\n",
                    )
                    selection_input = evidence_fs_prompt_truncated.format(claim=claim, passage=p_str)
                    example_num -= 1

                try:
                    cur_analysis = evidence_llm(selection_input).split('\n\n')[0].strip()
                    feedback = f"{cur_analysis}\nNow, extract evidence sentences from the analysis without any explanation. Each sentence should occupy one line and should be a complete sentence. If there is no evidence, please answer 'Not Enough Info'.\nEvidence Sentences:"
                    result = evidence_llm(evidence_fs_prompt_truncated.format(claim=claim, passage=p_str) + feedback).replace('<|im_end|>','').split('\n\n')[0].strip()
                    if 'not enough info' not in result.lower():
                        claim_evidence += result.split('\n')
                except:
                    continue
                claim_evidence = list(set(claim_evidence))
                
        final_evidence[claim] = claim_evidence

        feedback = f"\nNow, please extract the most relevant evidence sentences without any explanation. Each sentence should occupy one line and should be a complete sentence. Please copy the sentences to choose without altering their letter casing or punctuations. If there is no evidence, please answer 'Not Enough Info'.\nSelected Evidence Sentences:"
        final_selection_input = evidence_fs_prompt.format(claim=claim, passage='\n'.join(claim_evidence)) + feedback
        example_num = 9
        n_evidence = len(claim_evidence)
        while len(encoding.encode(final_selection_input)) > MAX_LENGTH - 256 and example_num > 0: 
            evidence_fs_prompt_truncated = FewShotPromptTemplate(
                examples=evidence_example[:example_num],
                example_prompt=evidence_prompt,
                prefix=evidence_selection_prefix,
                suffix="\nNow, let's focus on your task. You are given a claim and a passage. Please read the passage carefully and copy sentences that contain information supporting or refuting the claim.\n\nClaim: {claim}\nPassage:\n{passage}\nAnalysis:",
                input_variables=["claim", "passage"],
                example_separator="\n",
            )
            final_selection_input = evidence_fs_prompt_truncated.format(claim=claim, passage='\n'.join(claim_evidence)) + feedback
            example_num -= 1
        while len(encoding.encode(final_selection_input)) > MAX_LENGTH - 256 and n_evidence > 0:
            final_selection_input = evidence_fs_prompt_truncated.format(claim=claim, passage='\n'.join(claim_evidence[:n_evidence])) + feedback
            n_evidence -= 1
        
        final_analysis = evidence_llm(final_selection_input).replace('<|im_end|>','').split('\n\n')[0].strip()
        
        if 'not enough info' not in final_analysis.lower():
            final_evidence[claim] = final_analysis.split('\n')

        evidence_conclusion += f"{len(final_evidence[claim])} evidence sentences have been found for claim '{claim}'\n"
    
    return evidence_conclusion


def verdict_prediction(claims):
    global final_evidence
    global final_conclusion
    claims = claims.split('[EOS]')
    if claims != claim_splited:
        claims = claim_splited
    
    conclusions = []
    label_claims = {'refuted': [], 'not supported': [], 'partially supported': [], 'supported': []}
        
    for claim in claims:
        if claim not in final_evidence:
            return "Need to search for evidence first"
        evidence_list = final_evidence[claim]
        analysis = result_llm(verdict_prediction_fs_prompt.format(claim=claim, evidence='\n'.join(evidence_list))).split('\n\n')[0].strip()
        final_result_prompt = f"{analysis}\nBased on the conclusion, predict the final verdict label of the claim. The valid labels include 'supports', 'partially supports', 'not support', and 'refutes'.\nLabel:"
        subconclusion = result_llm(verdict_prediction_fs_prompt.format(claim=claim, evidence='\n'.join(evidence_list)) + '\n' + final_result_prompt).replace('_', ' ').split('\n\n')[0].strip().lower()
        
        if 'refute' in subconclusion:
            final_conclusion[claim] = 'refuted'
            label_claims['refuted'].append(claim)
            conclusions.append(f"'{claim}' is refuted")
        elif 'not support' in subconclusion:
            final_conclusion[claim] = 'not supported'
            label_claims['not supported'].append(claim)
            conclusions.append(f"'{claim}' is not supported")
        elif 'partially support' in subconclusion:
            final_conclusion[claim] = 'partially supported'
            label_claims['partially supported'].append(claim)
            conclusions.append(f"'{claim}' is partially supported")
        else:
            final_conclusion[claim] = 'supported'
            label_claims['supported'].append(claim)
            conclusions.append(f"'{claim}' is supported")
    
    label_conclusion = []
    for label in label_claims:
        if len(label_claims[label]) > 1:
            label_conclusion.append(f"{len(label_claims[label])} claims are {label}")
        elif len(label_claims[label]) == 1:
            label_conclusion.append(f"1 claim is {label}")
        
    conclusions.append(f"The input is decomposed into {len(final_conclusion)} subclaims, {', '.join(label_conclusion)}")
    return '\n'.join(conclusions)

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
        default='results/bingcheck',
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
        "--example_id",
        type=str,
        default=None,
        help="specific index",
    ) 
    parser.add_argument(
        "--split",
        action='store_true',
        help="split claim",
    )
    parser.add_argument(
        "--given_evidence",
        action='store_true',
        help="split claim",
    ) 
    args = parser.parse_args()

    return args

@tool("Claim_Processor", return_direct=False)
def claim_processor(input: str) -> str:
    """useful when processing the input response. the input should be the original language model response. if the input is too long, summarize the input without changing objective facts."""
    return claim_split(input)

@tool("Query_Writer", return_direct=False)
def query_writer(question: str) -> str:
    """useful for generating the search query to retrieve information to answer the user's question. the input should be the user question which the response answer to"""
    return generate_query(question)

@tool("Evidence_Seeker", return_direct=False)
def evidence_seeker(claims_queries: str) -> str:
    """
    useful for reading passages and finding evidence. the input must be all subclaims followed by all predicted search queries. the input must be in the format: subclaim_1[EOS]...[EOS]subclaim_n | query_1[EOS]...[EOS]query_n.
    """
    return select_evidence(claims_queries)

@tool("Verdict_Predictor", return_direct=False)
def verdict_predictor(claims: str) -> str:
    """
    useful for predicting the verdict label of a claim after obtaing evidence. the input should include all subclaims. the input must be in the format: subclaim_1[EOS]... [EOS]subclaim_n
    """
    return verdict_prediction(claims)

def main():
    global input_str
    global search_res
    global search_queries
    global claim_splited
    global final_conclusion
    global final_evidence

    args = parse_args()
    tools = [
            claim_processor,
            query_writer,
            evidence_seeker,
            verdict_predictor
        ]


    tool_introduction = '\n'.join([f"{tool.name}: {tool.description}" for tool in tools])
    actions = ', '.join([tool.name for tool in tools])
    tool_names = [tool.name for tool in tools]
    

    template = f"""Try your best to determine if the given input response is factually accurate. 
You have access to the following tools:

{tool_introduction}

Use the following format:

Input: the response of language model to the user query. you must verify the factual accuracy of the response.
Thought: you should always realize what you have known and think about what to do and which tool to use. 
Action: the action to take, should be one of [{actions}]
Action Input: the input to the action, must follow instructions of tools
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I can give an answer based on the evidence
Final Answer: a summary of verdict result of subclaims, such as the number of subclaims and the number of subclaims in each label class
"""
    template += "\n\nBelow is an example:\n\n"
    template += EXAMPLE
    template += "\nBegin!"
    template += "\n\nInput: {response}\n{agent_scratchpad}"


    complete_prompt = PromptTemplate(
        template=template,
        input_variables=["response","agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=llm_policy, prompt=complete_prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, prompt=complete_prompt)
    agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    result_file = {}
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(os.path.join(args.save_dir,args.save_file)):
        with open(os.path.join(args.save_dir,args.save_file)) as existing:
            result_file = json.load(existing)

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'bingcheck/examples.json'), "r", encoding="utf-8") as f:
        idx = 0
        data = json.load(f)
        for item_id,item in data.items():
            final_evidence = {}
            final_conclusion = {}
            search_res = {}
            claim_splited = []
            search_queries = []
            print(idx)
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
            input_str = input_response
            response = agent({'response':input_response})
            
            record = {
                'conclusion': response['output'].split('\n')[0].replace('<|im_end|>', ''),
                      }
            for step in response['intermediate_steps']:
                action, inter_result = step
                if action.tool == 'Claim_Processor':
                    record['subclaims'] = inter_result.split('[EOS]')
                if action.tool == 'Query_Writer':
                    record['queries'] = inter_result.strip().split('[EOS]')
            
            # record['search_results'] = search_res
            record['evidence'] = final_evidence
            record['subconclusions'] = final_conclusion
            
            if args.save_file:
                write_json(item['id'], record, os.path.join(args.save_dir, args.save_file))
                        

if __name__=='__main__':
    main()