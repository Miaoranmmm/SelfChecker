from langchain import PromptTemplate, FewShotPromptTemplate
import jsonlines
import os
prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_examples/wice')

EXAMPLES_SPLIT = f"""
Example_ID: train41647
Claim: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft, which revived the franchise with a new series under the same name.
Thought: I need to split the claim into subclaims before verify it. 
Action: Claim_Processor
Action Input: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft, which revived the franchise with a new series under the same name.
Observation: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft.
Ubisoft revived the franchise with a new series under the same name.
Thought: I need to verify subclaims based on retrieved information
Action: Evidence_Seeker
Action Input: train41647 | The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft.[EOS]Ubisoft revived the franchise with a new series under the same name.
Observation: 3 evidence sentences have been found for claim "The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft."
No evidence sentence has been found for claim "Ubisoft revived the franchise with a new series under the same name."
Thought: I need to determine the verdict label for each subclaim using the evidence.
Action: Verdict_Predictor
Action Input: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft.[EOS]Ubisoft revived the franchise with a new series under the same name.
Observation: "The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft." is supported.
"Ubisoft revived the franchise with a new series under the same name." is not supported.
Thought: I can give an answer based on the analysis for each subclaim
Final Answer: The claim is decomposed into 2 subclaims, 1 is supported and 1 is not supported. Therefore, the answer is partially supported.

Example_ID: train17534
Claim: Martin did important editorial work and co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Thought: I need to split the claim into subclaims before verify it. 
Action: Claim_Processor
Action Input: Martin did important editorial work and co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Observation: Martin did important editorial work.
Martin co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Thought: I need to verify the claim based on retrieved information
Action: Evidence_Seeker
Action Input: train17534 | Martin did important editorial work.[EOS]Martin co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Observation: No evidence sentence has been found for claim "Martin did important editorial work."
No evidence sentence has been found for claim "Martin co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966)."
Action: Verdict_Predictor
Action Input: Martin did important editorial work.[EOS]Martin co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Observation: "Martin did important editorial work." is not supported.
"Martin co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966)." is not supported.
Thought: I can give an answer based on the analysis
Final Answer: The claim is decomposed into 2 subclaims, both of them are not supported. Therefore, the answer is not supported
"""

EXAMPLES = f"""
Example_ID: train41647
Claim: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft, which revived the franchise with a new series under the same name.
Thought: I need to verify the claim based on retrieved information
Action: Evidence_Seeker
Action Input: train41647 | The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft, which revived the franchise with a new series under the same name.
Observation: The claim is partially supported by line 18 ; line 19 ; line 20
suggested answer: partially supported
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is partially supported

Example_ID: train17534
Claim: Martin did important editorial work and co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Thought: I need to verify the claim based on retrieved information
Action: Evidence_Seeker
Action Input: train17534 | Martin did important editorial work and co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Observation: The claim is not supported
suggested answer: not supported
Thought: I can give an answer based on the evidence
Final Answer: the claim is not supported
"""

EXAMPLES_SPLIT_EVIDENCE = f"""
Example_ID: train41647
Claim: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft, which revived the franchise with a new series under the same name.
Thought: I have a long claim to verify. I need to split the claim into subclaims before verify it. 
Action: Claim_Processor
Action Input: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft, which revived the franchise with a new series under the same name.
Observation: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft.
Ubisoft revived the franchise with a new series under the same name.
Thought: I need to decide whether subclaims are supported by given evidence
Action: Entailment_Classifier
Action Input: train41647 | The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft.[EOS]Ubisoft revived the franchise with a new series under the same name.
Observation: the claim is partially supported

Example_ID: train17534
Claim: Martin did important editorial work and co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Thought: I have a long claim to verify. I need to split the claim into subclaims before verify it. 
Action: Claim_Processor
Action Input: Martin did important editorial work and co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Observation: Martin did important editorial work.
Martin co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Thought: I need to decide whether subclaims are supported by given evidence
Action: Entailment_Classifier
Action Input: train17534 | Martin did important editorial work.[EOS]Martin co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Observation: the claim is not supported
"""

EXAMPLES_EVIDENCE = f"""
Example_ID: train41647
Claim: The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft, which revived the franchise with a new series under the same name.
Thought: I need to decide whether the claim is supported by given evidence
Action: Entailment_Classifier
Action Input: train41647 | The rights to the \"Might and Magic\" name were purchased for $1.3 million by Ubisoft, which revived the franchise with a new series under the same name.
Observation: the claim is partially supported

Example_ID: train17534
Claim: Martin did important editorial work and co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Thought: I need to decide whether the claim is supported by given evidence
Action: Entailment_Classifier
Action Input: train17534 | Martin did important editorial work and co-authored three influential books: \"Several complex variables\" (1948), \"Elementary differential equations\" (1956), and \"Differential space, quantum space, and prediction\" (1966).
Observation: the claim is not supported
"""

# Claim Split
split_prefix = """
You and your partners are on a mission to fact-check a complex claim that may contain multiple subclaims that need to be verified. A sentence that needs to be verified is any statement or assertion that requires evidence or proof to support its accuracy or truthfulness. For example, “Titanic was first released in 1997” necessitates verification of the accuracy of its release date, whereas a claim like "Water is wet" does not warrant verification. Each subclaim is a simple, complete sentence with single point to be verified. Imagine yourself as an expert in processing complex claim and extracting subclaims. Your task is to extract clear, unambiguous subclaims to check from the input paragraph, avoiding vague references like 'he,' 'she,' 'it,' or 'this,' and using complete names.

To illustrate the task, here are some examples:
"""

split_example_f = os.path.join(prompt_dir, 'split_example.jsonl')
split_example = []
with open(split_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        split_example.append(item)

split_template = """
Claim: {claim}
Subclaims: {subclaims}"""

split_prompt = PromptTemplate(
    input_variables=["claim","subclaims"],
    template=split_template,
)

split_fs_prompt = FewShotPromptTemplate(
    examples=split_example,
    example_prompt=split_prompt,
    prefix=split_prefix,
    suffix="\nNow, let's return to your task. You are given the following input paragraph, please extract all subclaims that need to be checked.\n\nClaim: {claim}\nSubclaims:",
    input_variables=["claim"],
    example_separator="\n",
)

# Evidence Selection
evidence_selection_prefix = '''
Your mission is to verify a claim's factual accuracy. As experts in reading comprehension, you'll receive a claim and a passage. Each line of the passage start with the line index.
You should first read the claim and the passage carefully. Make sure you understand what information you are looking for. Then select sentences that either support or partially support the claim. A sentence supports the claim if it provides evidence for all statements in the claim. A sentence partially supports the claim if it confirms some details but not all.
Exercise caution in your selection and judgment, avoiding overstatement. Choose the most relevant evidence and refrain from including noisy information. Base decisions solely on provided information without implying additional details.

To illustrate the task, here are some examples. Please learn from the examples and follow the template.
'''

evidence_template = """
Claim: {claim}
Passage:
{passage}
Analysis: {analysis}"""

evidence_prompt = PromptTemplate(
    input_variables=["claim", "passage", "analysis"],
    template=evidence_template,
)
evidence_example_f = os.path.join(prompt_dir, 'evidence_selection.jsonl')
evidence_example = []
with open(evidence_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        instance = {'claim': item['claim'], 'passage': item['passage'], 'analysis': item['analysis']}
        evidence_example.append(instance)

evidence_fs_prompt = FewShotPromptTemplate(
    examples=evidence_example,
    example_prompt=evidence_prompt,
    prefix=evidence_selection_prefix,
    suffix="\nNow, let's focus on your task. You are given a claim and a passage. Please read the passage carefully and copy the line index that contain information supporting the claim.\n\nClaims: {claim}\nPassage:\n{passage}\nAnalysis:",
    input_variables=["claim", "passage"],
    example_separator="\n",
)

# Verdict Prediction
verdict_prediction_prefix = '''
Your mission is to verify the factual accuracy of a claim using provided evidence. Your partners have collected evidence, and your expertise lies in assessing the claim's factualness based on this evidence.
You are required to determine whether the claim is supported, partially supported, or lacks sufficient information based on the provided evidence. The evidence supports the claim if it confirms all statements and details in the claim. The evidence partially supports the claim if it confirms some statements and details but not all. The evidence does not support the claim if it doesn't contradict or disprove any statement in the claim and doesn't support any statement.
Please exercise caution in making judgments and avoid overstatement. Base decisions solely on the provided information without implying additional details.

To illustrate the task, here are some examples. Please learn from the examples and follow the template.
'''

entailment_template = """
Claim: {claim}
Evidence: {evidence}
Analysis: {analysis}"""

verdict_prediction_prompt = PromptTemplate(
    input_variables=["claim", "evidence","analysis"],
    template=entailment_template,
)
entailment_example_f = os.path.join(prompt_dir, 'verdict_prediction.jsonl')
entailment_example = []
with open(entailment_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        entailment_example.append(item)

verdict_prediction_fs_prompt = FewShotPromptTemplate(
    examples=entailment_example,
    example_prompt=verdict_prediction_prompt,
    prefix=verdict_prediction_prefix,
    suffix="\nNow, let's return to your task. You are given the following claim and evidence. Please check whether the claim is supported, partially supported, or not supported by the given evidence.\n\nClaim: {claim}\nEvidence: {evidence}\nAnalysis:",
    input_variables=["claim", "evidence"],
    example_separator="\n",
)


# orca
B_INST, E_INST = "<im_start>", "<im_end>"
SYS, USR, AST = "system\n", "user\n", "assistant\n"
split_prefix_orca = """
You are a helpful assistant in extracting subclaims that need to be verified from a complex claim. 

A sentence that needs to be verified is any statement or assertion that requires evidence or proof to support its accuracy or truthfulness. 
For example, “Titanic was first released in 1997” necessitates verification of the accuracy of its release date, whereas a claim like "Water is wet" does not warrant verification. 

Each subclaim is a simple, complete sentence with single point to be verified. 
Please only provide the extraction result without other words.

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
"""
split_fs_prompt_orca = FewShotPromptTemplate(
    examples=split_example,
    example_prompt=split_prompt,
    prefix='<im_start>system'+split_prefix_orca,
    suffix="\n<im_end>\n\n<im_start>user\nClaim: {claim}\n<im_end>\n<im_start>assistant\nSubclaims:",
    input_variables=["claim"],
    example_separator="\n",
)

evidence_selection_prefix_orca = '''
You are an experts in reading comprehension. You'll receive a claim and a passage. Each line of the passage start with the line index.
You should first read the claim and the passage carefully. Make sure you understand what information you are looking for. Then select sentences that either support or partially support the claim. A sentence supports the claim if it provides evidence for all statements in the claim. A sentence partially supports the claim if it confirms some details but not all.
Exercise caution in your selection and judgment, avoiding overstatement. Choose the most relevant evidence and refrain from including noisy information. Base decisions solely on provided information without implying additional details.

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
'''
evidence_fs_prompt_orca = FewShotPromptTemplate(
    examples=evidence_example,
    example_prompt=evidence_prompt,
    prefix='<im_start>system'+evidence_selection_prefix_orca,
    suffix="\n<im_end>\n\n<im_start>user\nClaims: {claim}\nPassage:\n{passage}\n<im_end>\n<im_start>assistant\nAnalysis:",
    input_variables=["claim", "passage"],
    example_separator="\n",
)

verdict_prediction_prefix_orca = '''
You are an expert in verifying the factual accuracy of a claim using provided evidence.
You are required to determine whether the given claim is supported, partially supported, or lacks sufficient information based on the provided evidence. The evidence supports the claim if it confirms all statements and details in the claim. The evidence partially supports the claim if it confirms some statements and details but not all. The evidence does not support the claim if it doesn't contradict or disprove any statement in the claim and doesn't support any statement.
Please exercise caution in making judgments and avoid overstatement. Base decisions solely on the provided information without implying additional details.

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
'''
verdict_prediction_fs_prompt_orca = FewShotPromptTemplate(
    examples=entailment_example,
    example_prompt=verdict_prediction_prompt,
    prefix='<im_start>system'+verdict_prediction_prefix_orca,
    suffix="\n<im_end>\n\n<im_start>user\nClaim: {claim}\nEvidence: {evidence}\n<im_end>\n<im_start>assistant\nAnalysis:",
    input_variables=["claim", "evidence"],
    example_separator="\n",
)


# llama
split_prefix_llama = """
You are a helpful assistant in extracting subclaims that need to be verified from a complex claim. 

A sentence that needs to be verified is any statement or assertion that requires evidence or proof to support its accuracy or truthfulness. 
For example, “Titanic was first released in 1997” necessitates verification of the accuracy of its release date, whereas a claim like "Water is wet" does not warrant verification. 

Please only provide the extraction result without any other word.
Each subclain is in one separate line. Do not add any indicator sign before each claim.

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
"""
split_fs_prompt_llama = FewShotPromptTemplate(
    examples=split_example,
    example_prompt=split_prompt,
    prefix='[INST] <<SYS>>'+split_prefix_llama,
    suffix="<</SYS>>\nNow start working on the following claim.\nClaim: {claim}\n[/INST]Subclaims:",
    input_variables=["claim"],
    example_separator="\n",
)

evidence_selection_prefix_llama = '''
You are an experts in reading comprehension. You'll receive a claim and a passage. Each line of the passage start with the line index.
You should first read the claim and the passage carefully. Make sure you understand what information you are looking for. Then select sentences that either support or partially support the claim. A sentence supports the claim if it provides evidence for all statements in the claim. A sentence partially supports the claim if it confirms some details but not all.
Exercise caution in your selection and judgment, avoiding overstatement. Only choose the most relevant evidence and refrain from including noisy information. Base decisions solely on provided information without implying additional details.

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
'''

evidence_fs_prompt_llama = FewShotPromptTemplate(
    examples=evidence_example,
    example_prompt=evidence_prompt,
    prefix='[INST] <<SYS>>'+evidence_selection_prefix_llama,
    suffix="<</SYS>>\nNow, please focus on your task. You are given a claim and a passage. Please read the passage carefully and copy the line index that contain information supporting the claim..\nClaim: {claim}\nPassage:\n{passage}\n[/INST]",
    input_variables=["claim", "passage"],
    example_separator="\n",
)

verdict_prediction_prefix_llama = '''
You are an expert in verifying the factual accuracy of a claim using provided evidence.
You are required to determine whether the given claim is supported, partially supported, or lacks sufficient information based on the provided evidence. The evidence supports the claim if it confirms all statements and details in the claim. The evidence partially supports the claim if it confirms some statements and details but not all. The evidence does not support the claim if it doesn't contradict or disprove any statement in the claim and doesn't support any statement.
Please exercise caution in making judgments and avoid overstatement. Base decisions solely on the provided information without implying additional details.

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
'''
verdict_prediction_fs_prompt_llama = FewShotPromptTemplate(
    examples=entailment_example,
    example_prompt=verdict_prediction_prompt,
    prefix='[INST] <<SYS>>'+verdict_prediction_prefix_llama,
    suffix="<</SYS>>\nNow analyze the following claim and evidence.\nClaim: {claim}\nEvidence: {evidence}\n[/INST]",
    input_variables=["claim", "evidence"],
    example_separator="\n",
)