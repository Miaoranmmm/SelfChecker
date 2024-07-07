from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
import os
import re
import jsonlines
import json
import argparse
import sqlite3
from retry import retry
from collections import Counter
from utils import *
from dotenv import load_dotenv
load_dotenv()
os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://llmaugmenter.openai.azure.com/" # "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"#"2023-05-15"

TEMPERATURE = 0.2
DEPLOYMENT_NAME = "text003"
MODEL_NAME = "text003"
llm = AzureOpenAI(deployment_name=DEPLOYMENT_NAME, model_name=MODEL_NAME, n=1, temperature=TEMPERATURE)

prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt/prompt_examples')

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--save_file",
        type=str,
        default='results_standard_prompt.json',
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
        default='prompts/prompt_exmples/fever/standard_prompt.jsonl',
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

    example_template = """
Claim: {claim}
Label: {label}"""

    prefix = f"""Imagine you are an expert in determining the factualness of claims. Please use your knowledge and determine the verdict label for the given claim. The candidate labels can be {args.verdict_labels}.

Learn from the examples below:
"""

    example_prompt = PromptTemplate(
        input_variables=["claim","label"],
        template=example_template,
    )
    example_f = os.path.join(prompt_dir, args.example_file)
    icl_examples = []
    with open(example_f, "r", encoding="utf-8") as reader:
        for item in jsonlines.Reader(reader):
            # item['wiki_name'] = item['wiki_name']
            icl_examples.append(item)


    fs_prompt = FewShotPromptTemplate(
        examples=icl_examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="\nNow, let's return to your task. You are given the following claim, please predict the verdict label for it.\n\nClaim: {claim}\nLabel:",
        input_variables=["claim"],
        example_separator="\n",
    )

    # print(complete_prompt.format(claim="big",agent_scratchpad=""))
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
        print(len(result_file))
    elif file_name.endswith('jsonl'):
        output_file = jsonlines.open(os.path.join(args.save_dir, args.save_file), 'w')
        result_file = []
    elif file_name.endswith('json'):
        result_file = {}

    if args.eval_dataset == 'fever':
        with open('fever/sampled_ids.json') as sample_file:
            sampled_ids = json.load(sample_file)
            print(f'# sampled_ids: {len(sampled_ids)}')

        with open(args.test_file, "r", encoding="utf-8") as reader:
            idx = 0
            for item in jsonlines.Reader(reader):
                # print(idx)
                if args.sampled and int(item['id']) not in sampled_ids:
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

                response = llm(fs_prompt.format(claim=item['claim']))['output']
    
                print(item['label'])
                record = {
                    'claim': item['claim'],
                    'conclusion': response,
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

                input_response = re.sub('\[\^.*?\^\]', '', item['response'].replace('\n',' ').replace('  ',' ').replace('**',''))
                response = llm(fs_prompt.format(claim=input_response)).lower()
                analysis = response.split('\n')[0].replace('<|im_end|>', '')
                print(response)
                if 'partially support' in response:
                    predicted_label = 'PARTIALLY SUPPORT'
                elif 'not support' in response:
                    predicted_label = 'NOT SUPPORT'
                elif 'refute' in response:
                    predicted_label = 'REFUTE'
                elif 'support' in response:
                    predicted_label = 'SUPPORT'
                else:
                    predicted_label = 'NOT SUPPORT'
                print(predicted_label)
                
                record = {'analysis':'', 'conclusion':predicted_label}

                if args.save_file:
                    write_json(item['id'], record, os.path.join(args.save_dir, args.save_file))
                print(idx)
                idx += 1
                print('='*60)
                
    elif args.eval_dataset == 'wice':
        with open(args.test_file, "r", encoding="utf-8") as reader:
            idx = 0
            for item in jsonlines.Reader(reader):
                # print(idx)
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
                    if item['meta']['id'] != args.example_id:
                        continue

                response = llm(fs_prompt.format(claim=item['claim']))
                print(response)
                print()
                predicted_label = response
                if 'partially support' in predicted_label.lower():
                    predicted_label = 'PARTIALLY SUPPORT'
                elif 'not support' in predicted_label.lower():
                    predicted_label = 'NOT SUPPORT'
                elif 'refute' in predicted_label.lower():
                    predicted_label = 'REFUTE'
                elif 'support' in predicted_label.lower():
                    predicted_label = 'SUPPORT'
                else:
                    predicted_label = 'NOT SUPPORT'
                print(predicted_label)
                print(item['label'])
                record = {'analysis':'', 'conclusion':predicted_label}
                
                if args.save_file:
                    write_json(item['meta']['id'], record, os.path.join(args.save_dir, args.save_file))
                print(idx)
                idx += 1
                print('='*60)


if __name__=='__main__':
    main()