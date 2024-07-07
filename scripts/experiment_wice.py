from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor, tool, initialize_agent
from langchain.prompts import PromptTemplate
import os
import jsonlines
import json
import argparse
import re
import sqlite3
from retry import retry
from collections import Counter
from prompts.prompt_wice import *
from utils import *
from dotenv import load_dotenv
load_dotenv()
os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://llmaugmenter.openai.azure.com/" # "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"#"2023-05-15"

regex = r'\'(.*)\''
claim_regex = r'the claim \'(.*)\' is'
punctuation = r'[,.;@#?!&$]+\ *'

TEMPERATURE = 0.2
DEPLOYMENT_NAME = "text003"
MODEL_NAME = "text003"

split_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE) #deployment_name model_name #"chatgpt" "text003" "gpt-35-turbo"
evidence_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE) # temp 0.2
result_llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE) # temp 0.2
llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)

claim_splited = []
final_evidence = {}

## Claim Split ##
def claim_split(claim):
    global claim_splited
    result = split_llm(split_fs_prompt.format(claim=claim)).replace('<|im_end|>', '').strip()
    claim_splited = result.split('\n')
    claim_splited = [c for c in claim_splited if c.strip() != '']
    return result

db = 'wice_passages.db'
conn = sqlite3.connect(db)
c = conn.cursor()
def querydb(example_id, like_search=False):
    name = example_id
    lines_query = f"SELECT * from passages WHERE id='{name}'"
    c.execute(lines_query)
    lines = c.fetchone()
    if not lines and like_search:
        lines_query = f"SELECT * from passages WHERE id LIKE '%{name}%'"
        c.execute(lines_query)
        lines = c.fetchone()
    return lines


## Evidence Selection ##
def select_evidence(claim_id):
    global claim_splited 
    global final_evidence
    evidence_conclusion = ''
    
    example_id, claims = claim_id.split(' | ')
    retrieved_passage = querydb(example_id)[1]
    lines = retrieved_passage.split('\n')
    n_lines = len(lines)
    conclusion = 'not supported'
    evidence = 'no evidence'
    
    selected_evidences = []

    claims = claims.split('[EOS]')
    if claims != claim_splited:
        if len(claims) > len(claim_splited):
            claim_splited = claims
        else:
            claims = claim_splited

    for claim in claims:
        analysis = []
        evidence_list = []
        candidate_evidence = []
        for i in range(0, n_lines, 10):
            passage = '\n'.join(lines[i: min(n_lines, i+10)])
            try:
                cur_analysis = evidence_llm(evidence_fs_prompt.format(claim=claim, passage=passage)).split('\n\n')[0].strip()
                analysis.append(cur_analysis)
            except:
                continue
           
            cur_analysis_list = cur_analysis.split('\n')
            for cur_analysis in cur_analysis_list:
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
                        sent = '\t'.join(lines[int(s_idx)].strip().split('\t')[1:])
                        if len(sent) > 0:
                            selected_evidences.append(sent)
                            candidate_evidence.append(f"{s_idx}\t{sent}")
                except:
                    pass

        candidate_evidence_str = '\n'.join(candidate_evidence) + '\n'

        final_analysis = evidence_llm(evidence_fs_prompt.format(claim=claim, passage=candidate_evidence_str)).split('\n\n')[0].strip().replace(' lines ',' line ')
        final_analysis_list = final_analysis.split('\n')
        for final_analysis in final_analysis_list:
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
                    sent = '\t'.join(lines[int(s_idx)].strip().split('\t')[1:])
                    if len(sent) > 0:
                        evidence_list.append(f'{s_idx}\t{sent}')
               
            except:
                continue
        
        if  len(evidence_list) < 1:
            evidence_list = candidate_evidence

        final_evidence[claim] = evidence_list
        evidence_conclusion += f"{len(final_evidence[claim])} evidence sentences have been found for claim '{claim}'\n"
    
    return evidence_conclusion

def verdict_prediction(claims):
    global final_evidence
    global final_conclusion
    claims = claims.split('[EOS]')
    if claims != claim_splited:
        claims = claim_splited

    conclusions = []
    label_claims = {'not supported': [], 'partially supported': [], 'supported': []}
        
    for claim in claims:
        if claim not in final_evidence:
            return "Need to search for evidence first"
        evidence_list = final_evidence[claim]
        evidence_list_processed = ['\t'.join(evd_piece.split('\t')[1:]) for evd_piece in evidence_list]
        analysis = result_llm(verdict_prediction_fs_prompt.format(claim=claim, evidence='\n'.join(evidence_list_processed))).split('\n\n')[0].strip()
        final_result_prompt = f"{analysis}\nBased on the conclusion, predict the final verdict label of the claim. The valid labels include 'support', 'partially support', and 'not support'.\nLabel:"
        subconclusion = result_llm(verdict_prediction_fs_prompt.format(claim=claim, evidence='\n'.join(evidence_list)) + '\n' + final_result_prompt).replace('_', ' ').split('\n\n')[0].strip().lower()
        
        if 'not support' in subconclusion:
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
        default='results/wice',
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

def main():
    global claim_splited
    global final_evidence
    global final_conclusion
    args = parse_args()

    @tool("Claim_Processor", return_direct=False)
    def claim_processor(claim: str) -> str:
        """useful when splitting a claim into simple subclaims. should use it after obtaining useful information. the input must be the original long claim"""
        return claim_split(claim)

    @tool("Evidence_Seeker", return_direct=False)
    def evidence_seeker(claims_id: str) -> str:
        """
        useful for reading passages and finding evidence. the input must be the example id and all subclaims and in the format: example_id | subclaim_1[EOS]... [EOS]subclaim_n
        """
        return select_evidence(claims_id)

    @tool("Verdict_Predictor", return_direct=False)
    def verdict_predictor(claims: str) -> str:
        """
        useful for predicting the verdict label of a claim after obtaing evidence. the input should include all subclaims. the input must be in the format: subclaim_1[EOS]... [EOS]subclaim_n
        """
        return verdict_prediction(claims)

    if args.split: 
        # case 1: split + evidence_seeking
        if not args.given_evidence:  
            tools = [
                claim_processor,
                evidence_seeker,
                verdict_predictor
            ]
            thought = (
            '1. split the claim into several simple subclaims'
            '2. after getting subclaims and example id, try to find evidence lines that supports or partially supports the subclaims from the passage. Make sure the input format is correct.'
            '3. after retrieving evidence, determine the verdict label of the subclaims'
            'you can go back to previous steps if you need to find more information'
            )
        
        # case 2: split + verdict prediction
        else:
            tools = [
                claim_processor,
                verdict_predictor
            ]
            thought = (
            '1. split the claim into several simple subclaims'
            '2. after retrieving evidence, determine the verdict label of the subclaims'
            'you can go back to previous steps if you need to find more information'
            )        
    
    else:
        # case 3: no split only evidence seeking
        if not args.given_evidence: 
            tools = [
                evidence_seeker,
                verdict_predictor
            ]
            thought = (
            '1. after getting the example id, try to find evidence lines that supports or partially supports the claim from the passage. Make sure the input format is correct.'
            '2. after retrieving evidence, determine the verdict label of the claim'
            'you can go back to previous steps if you need to find more information'
            )

        # case 4: no split only verdict prediction
        else: 
            tools = [
                verdict_predictor
            ]
            thought = (
            '1. after retrieving evidence, determine the verdict label of the claim'
            'you can repeat the step if you need to find more information'
            )
            

    tool_introduction = '\n'.join([f"{tool.name}: {tool.description}" for tool in tools])
    actions = ', '.join([tool.name for tool in tools])
    tool_names = [tool.name for tool in tools]
    

    template = f"""Try your best to determine if the given claim is supported, partially supported or not supported. Normally you should follow the steps: {thought}
You have access to the following tools:

{tool_introduction}

Use the following format:

Example_ID: example id 
Claim: the claim you must verify
Thought: you should always realize what you have known and think about what to do and which tool to use. 
Action: the action to take, should be one of [{actions}]
Action Input: the input to the action, must follow instructions of tools
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I can give an answer based on the evidence
Final Answer: should be in the form: supported, partially supported or not supported

Learn from the examples below:
"""

    if args.split and not args.given_evidence:
        template += EXAMPLES_SPLIT
    elif args.split and args.given_evidence:
        template += EXAMPLES_SPLIT_EVIDENCE
    elif not args.split and args.given_evidence:
        template += EXAMPLES_EVIDENCE
    else:
        template += EXAMPLES
    
    template += "\nBegin!"
    template += "\n\nExample ID: {example_id}\nClaim: {claim}\n{agent_scratchpad}"


    complete_prompt = PromptTemplate(
        template=template,
        input_variables=["claim","example_id","agent_scratchpad"]
    )

    
    llm_chain = LLMChain(llm=llm, prompt=complete_prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, prompt=complete_prompt)
    agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    
    result_file = {}
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if os.path.exists(os.path.join(args.save_dir,args.save_file)):
        with open(os.path.join(args.save_dir,args.save_file)) as existing:
            result_file = json.load(existing)

    with jsonlines.open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wice/data/entailment_retrieval/claim/test.jsonl")) as reader:
        idx = len(result_file)
        for item in reader:
            claim_splited = []
            final_evidence = {}
            final_conclusion = {}
            item_id = item['meta']['id']
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
            elif item_id in result_file:
                idx += 1
                continue
            
            print('CLAIM:', item['claim'])

            if args.split and args.given_evidence:  
                passage = querydb(item_id)[1]
                with jsonlines.open (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wice/data/entailment_retrieval/subclaim/test.jsonl")) as subreader:
                    for subitem in subreader:
                        subclaim_id = subitem['meta']['id']
                        subclaim = subitem['claim']
                        if subclaim_id.split('-')[0] == item_id:
                            evd_index  = []
                            for evds in subitem['supporting_sentences']:
                                evd_index += evds
                            evd_index = list(set(evd_index))
                            evidence_list = [passage[evd_id] for evd_id in evd_index]
                            final_evidence[subclaim] = evidence_list
                        else:
                            continue
            
            elif not args.split and args.given_evidence:
                # case: not split only verdict prediction
                evd_index  = []
                for evds in item['supporting_sentences']:
                    evd_index += evds
                evd_index = list(set(evd_index))
                passage = querydb(item_id)[1]
                evidence_list = [passage[evd_id] for evd_id in evd_index]
                final_evidence[claim] = evidence_list
                
            # response = agent({'claim':item['claim'], 'example_id': item_id})
            try:
                response = agent({'claim':item['claim'], 'example_id': item_id})
            except:
                idx += 1
                continue
            conclusion = response['output'].lower()
            if 'partially supported' in conclusion:
                conclusion = 'partially supported'
            elif 'not supported' in conclusion:
                conclusion = 'not supported'
            elif 'supported' in conclusion:
                conclusion = 'supported'
            
            print(item['label'])
            supporting_sentences = item['supporting_sentences']
            gold_evi = []
            for sentences in supporting_sentences:
                gold_evi += sentences
            gold_evi = list(set(gold_evi))
            record = {
                'claim': item['claim'],
                'conclusion': conclusion,
                'label': item['label'],
                'gold_evi': gold_evi
            }
            subclaim_records = []
            if len(claim_splited) > 0:
                for subclaim in claim_splited:
                    if subclaim not in final_conclusion:
                        subclaim_record = {
                            'subclaim': subclaim, 
                            'evidence': [],
                            'conclusion': 'not supported'
                        }
                    else:
                        if final_conclusion[subclaim] == 'not supported':
                            subclaim_evd = []
                        else:
                            subclaim_evd = [int(evd_piece.split('\t')[0]) for evd_piece in final_evidence[subclaim]]
                        subclaim_record = {
                            'subclaim': subclaim, 
                            'evidence': subclaim_evd,
                            'conclusion': final_conclusion[subclaim]
                        }
                    subclaim_records.append(subclaim_record)
            record['subclaim_results'] = subclaim_records

            result_file[item_id] = record
            
            if args.save_file:
                write_json(item_id, record, os.path.join(args.save_dir,args.save_file))  


if __name__=='__main__':
    main()