import os
import jsonlines
from langchain import PromptTemplate, FewShotPromptTemplate

prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_examples/bingcheck')


EXAMPLE=f'''
Response: The Fibonacci sequence is a series of numbers where each number is the sum of the two previous ones. The Fibonacci sequence has golden ratio, which is approximately equal to 1.618 and can be found in many applications. In music, some composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piecesome composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piece. In architecture, some famous examples of buildings that incorporate the golden ratio are the Parthenon in Greece, the Taj Mahal in India, and the Notre Dame Cathedral in France.
Thought: I should first process the response I need to verify
Action: Claim_Processor
Action Input: The Fibonacci sequence is a series of numbers where each number is the sum of the two previous ones. The Fibonacci sequence has golden ratio, which is approximately equal to 1.618 and can be found in many applications. In music, some composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piecesome composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piece. In architecture, some famous examples of buildings that incorporate the golden ratio are the Parthenon in Greece, the Taj Mahal in India, and the Notre Dame Cathedral in France.
Observation: The Fibonacci sequence is a series of numbers where each number is the sum of the two previous ones.
Some composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piece.
The Parthenon in Greece, the Taj Mahal in India, and the Notre Dame Cathedral in France incorporate the golden ratio in their design.
Thought: I need to predict search queries for the subclaims to retrieve information
Action: Query_Writer
Action Input: The Fibonacci sequence is a series of numbers where each number is the sum of the two previous ones.[EOS]Some composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piece.[EOS]The Parthenon in Greece, the Taj Mahal in India, and the Notre Dame Cathedral in France incorporate the golden ratio in their design.
Observation: Fibonacci sequence definition and properties[EOS]Composers using Fibonacci numbers in music composition[EOS]Parthenon Taj Mahal Notre Dame golden ratio design
Thought: I have search queries for all subclaims. Now I need to find evidence for each subclaim using the search queries.
Action: Evidence_Seeker
Action Input: The Fibonacci sequence is a series of numbers where each number is the sum of the two previous ones.[EOS]Some composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piece.[EOS]The Parthenon in Greece, the Taj Mahal in India, and the Notre Dame Cathedral in France incorporate the golden ratio in their design. | Fibonacci sequence definition and properties[EOS]Composers using Fibonacci numbers in music composition[EOS]Parthenon Taj Mahal Notre Dame golden ratio design
Observation: 5 evidence sentences have been found for claim 'The Fibonacci sequence is a series of numbers where each number is the sum of the two previous ones.'
7 evidence sentences have been found for claim 'Some composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piece.'
4 evidence sentences have been found for claim 'The Parthenon in Greece, the Taj Mahal in India, and the Notre Dame Cathedral in France incorporate the golden ratio in their design.'
Thought: I need to determine the verdict label for each subclaim using the evidence.
Action: Verdict_Predictor
Action Input: The Fibonacci sequence is a series of numbers where each number is the sum of the two previous ones.[EOS]Some composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piece.[EOS]The Parthenon in Greece, the Taj Mahal in India, and the Notre Dame Cathedral in France incorporate the golden ratio in their design.
Observation: 'The Fibonacci sequence is a series of numbers where each number is the sum of the two previous ones.' is supported
'Some composers have used the Fibonacci numbers to determine the length of notes or the number of bars in a piece.' is supported
'The Parthenon in Greece, the Taj Mahal in India, and the Notre Dame Cathedral in France incorporate the golden ratio in their design.' is supported
Thought: I can give an answer based on the analysis for each subclaim
Final Answer: The input is decomposed into 3 subclaims, all of them are supported.
'''


## Claim Processing ##
split_example_f = os.path.join(prompt_dir, 'claim_split.jsonl')
split_example = []
with open(split_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        split_example.append(item)

split_template = """
Input: {input}
Subclaims: {subclaims}"""

split_prompt = PromptTemplate(
    input_variables=["input","subclaims"],
    template=split_template,
)

claim_processor_prefix = """
You and your partners are on a mission to fact-check a paragraph that may contain multiple subclaims that need to be verified. A sentence that needs to be verified is any statement or assertion that requires evidence or proof to support its accuracy or truthfulness. For example, “Titanic was first released in 1997” necessitates verification of the accuracy of its release date, whereas a claim like "Water is wet" does not warrant verification. Each subclaim is a simple, complete sentence with single point to be verified. Imagine yourself as an expert in processing complex paragraphs and extracting subclaims. Your task is to extract clear, unambiguous subclaims to check from the input paragraph, avoiding vague references like 'he,' 'she,' 'it,' or 'this,' and using complete names.

To illustrate the task, here are some examples:
"""

split_fs_prompt = FewShotPromptTemplate(
    examples=split_example,
    example_prompt=split_prompt,
    prefix=claim_processor_prefix,
    suffix="\nNow, let's return to your task. You are given the following input paragraph, please extract all subclaims that need to be checked.\n\nInput: {input}\nSubclaims:",
    input_variables=["input"],
    example_separator="\n",
)

## Query Generation ##
query_template = """
Context: {context}
Claim: {claim}
Query: {query}"""

query_prompt = PromptTemplate(
    input_variables=["context","claim","query"],
    template=query_template,
)
query_example_f = os.path.join(prompt_dir, 'query_generation.jsonl')
query_example = []
with open(query_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        query_example.append(item)

query_prompt_prefix = """
You and your partners are on a mission to fact-check a paragraph. Subclaims requiring verification have been extracted from the paragraph. Imagine yourself as an internet research expert. Your task is to generate a search query for each subclaim to find relevant information for fact-checking. You will be provided with the context of a claim and the specific claim for which you should create a search query.

To illustrate the task, here are some examples:
"""
query_fs_prompt = FewShotPromptTemplate(
    examples=query_example,
    example_prompt=query_prompt,
    prefix=query_prompt_prefix,
    suffix="\nNow, let's return to your task. You are given the following claim and its context, please predict the most appropriate search query for it.\n\nContext: {context}\nClaim: {claim}\nQuery:",
    input_variables=["context","claim"],
    example_separator="\n",
)

## Evidence Seeker ##
evidence_template = """
Claim: {claim}
Passage:
{passage}
Analysis: {analysis}"""

evidence_prompt = PromptTemplate(
    input_variables=["claim", "passage", "analysis"],
    # input_variables=["claims", "passage", "evidence"],
    template=evidence_template,
)
evidence_example_f = os.path.join(prompt_dir, 'evidence_selection.jsonl')
# evidence_example_f = 'bing_chat_example/context_example_new.jsonl'
evidence_example = []
with open(evidence_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        # instance = {'claims': item['claims'], 'passage': item['passage'], 'analysis': item['analysis']}
        evidence_example.append(item)

evidence_selection_prefix = '''
Your mission is to verify a claim's factual accuracy. As experts in reading comprehension, you'll receive a claim and a passage.
You should first read the claim and the passage carefully. Make sure you understand what information you are looking for. Then select sentences that either support, partially support, or refute the claim. A sentence supports the claim if it provides evidence for all statements in the claim. A sentence partially supports the claim if it confirms some details but not all. A sentence refutes the claim if it contradicts any statement in the claim.
Exercise caution in your selection and judgment, avoiding overstatement. Choose the most relevant evidence and refrain from including noisy information. Base decisions solely on provided information without implying additional details.

To illustrate the task, here are some examples:
'''

evidence_fs_prompt = FewShotPromptTemplate(
    examples=evidence_example,
    example_prompt=evidence_prompt,
    prefix=evidence_selection_prefix,
    suffix="\nNow, let's focus on your task. You are given a claim and a passage. Please read the passage carefully and copy sentences that contain information supporting or refuting the claim.\n\nClaim: {claim}\nPassage:\n{passage}\nAnalysis:",
    input_variables=["claim", "passage"],
    example_separator="\n",
)

# Entailment
entailment_template = """
Claim: {claim}
Evidence: {evidence}
Analysis: {analysis}"""

verdict_prediction_prefix = '''
Your mission is to verify the factual accuracy of a claim using provided evidence. Your partners have collected evidence, and your expertise lies in assessing the claim's factualness based on this evidence.
You must determine if the claim is supported, partially supported, refuted, or not supported based on the provided evidence. The evidence supports the claim if it confirms all statements and details in the claim. The evidence partially supports the claim if it confirms some statements and details but not all. The evidence refutes the claim if it contradicts any statement in the claim. The evidence does not support the claim if it doesn't contradict or disprove any statement in the claim and doesn't support any statement.
Please exercise caution in making judgments and avoid overstatement. Base decisions solely on the provided information without implying additional details.

Here are examples to illustrate the task:
'''

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
    suffix="\nNow, let's return to your task. You are given the following claim and evidence. Please check whether the claim is supported, partially supported, not supported or refuted by the given evidence.\n\nClaim: {claim}\nEvidence: {evidence}\nAnalysis:",
    input_variables=["claim", "evidence"],
    example_separator="\n",
)
