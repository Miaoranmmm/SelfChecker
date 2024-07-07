from langchain import PromptTemplate, FewShotPromptTemplate
import jsonlines
import os
prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_examples/fever')

EXAMPLES_FULL_PIPELINE = f"""
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought: I have a claim to verify. I need to first generate search queries to find more information
Action: QueryWriter
Action Input: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Observation: Nikolaj_Coster-Waldau ; Fox_Broadcasting
Thought: I have known candidate search queries. I need to use them to obtain names of passages from the database 
Action: Database
Action Input: Nikolaj_Coster-Waldau ; Fox_Broadcasting
Observation: Nikolaj_Coster-Waldau ; Fox_Broadcasting_Company
Thought: I now know names of passages to read. I should find evidence sentences from passages. I should input in the format: claim | names of passages
Action: EvidenceSeeker
Action Input: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. | Nikolaj_Coster-Waldau ; Fox_Broadcasting_Company
Observation: <Nikolaj_Coster-Waldau> line 7: He then played Detective John Amsterdam in the short-lived Fox television series New Amsterdam -LRB- 2008 -RRB- , as well as appearing as Frank Pike in the 2009 Fox television film Virtuality , originally intended as a pilot . Fox television Fox Broadcasting Company New Amsterdam New Amsterdam (TV series) Virtuality Virtuality (TV series) | <Fox_Broadcasting_Company> line 0: The Fox Broadcasting Company -LRB- often shortened to Fox and stylized as FOX -RRB- is an American English language commercial broadcast television network that is owned by the Fox Entertainment Group subsidiary of 21st Century Fox . English English language Fox Entertainment Group Fox Entertainment Group 21st Century Fox 21st Century Fox American United States English language English language commercial commercial broadcasting broadcast terrestrial television television network television network
Thought: I have found evidence. I need to determine the verdict label the claim.
Action: VerdictPredictor
Action Input: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. | <Nikolaj_Coster-Waldau> line 7 ; <Fox_Broadcasting_Company> line 0
Observation: The evidence supports the claim because it provides clear and direct information that Nikolaj Coster-Waldau played the role of Detective John Amsterdam in the Fox television series New Amsterdam in 2008 and also appeared as Frank Pike in the 2009 Fox television film Virtuality. This information directly establishes his work with the Fox Broadcasting Company, confirming the claim's validity.
The answer is supports
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is supported

Claim: Stranger Things is set in Bloomington, Indiana.
Thought: I need to generate search query to find more information
Action: QueryWriter
Action Input: Stranger Things is set in Bloomington, Indiana.
Observation: Stranger_Things
Thought: I have known candidate search queries. I need to use them to obtain names of passages from the database
Action: Database
Action Input: Stranger_Things
Observation: Stranger_Things
Thought: I now know names of passages to read. I should find evidence sentences from passages. I should input in the format: claim | names of passagess
Action: EvidenceSeeker
Action Input: Stranger Things is set in Bloomington, Indiana. | Stranger_Things
Observation: <Stranger_Things> line 5: Set in the fictional town of Hawkins , Indiana in the 1980s , the first season focuses on the investigation into the disappearance of a young boy by his friends , older brother and traumatized mother and the local police chief , amid supernatural events occurring around the town including the appearance of a psychokinetic girl who helps the missing boy 's friends in their own search .\tpsychokinetic\tpsychokinesis
Thought: I have found evidence. I need to determine the verdict label for the claim.
Action: VerdictPredictor
Action Input: Stranger Things is set in Bloomington, Indiana. | <Stranger_Things> line 5
Observation: The evidence refutes the claim because it specifies that 'Stranger Things' is set in the fictional town of Hawkins, Indiana, which implies it is not set in Bloomington, Indiana.
The answer is refutes
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is refuted

Claim: System of a Down briefly disbanded in limbo.
Thought: I need to generate search query to find more information
Action: QueryWriter
Action Input: System of a Down briefly disbanded in limbo.
Observation: limbo
Thought: I have known candidate search queries. I need to search the database for passage names
Action: Database
Action Input: limbo
Observation: Title_in_Limbo
Thought: I now know names of passages to read. I should find evidence sentences from passages. I should input in the format: claim | names of passagess
Action: EvidenceSeeker
Action Input: System of a Down briefly disbanded in limbo. | Title_in_Limbo
Observation: no evidence
Thought: I can give an answer now
Final Answer: there is not enough information
"""


EXAMPLES_GOLD_PASSAGE = """
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Passage Names: Nikolaj_Coster-Waldau ; Fox_Broadcasting_Company
Thought: I should find evidence sentences from passages. I should input in the format: claim | names of passagess
Action: Evidence_Seeker
Action Input: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. | Nikolaj_Coster-Waldau ; Fox_Broadcasting_Company
Observation: <Nikolaj_Coster-Waldau> line 7: He then played Detective John Amsterdam in the short-lived Fox television series New Amsterdam -LRB- 2008 -RRB- , as well as appearing as Frank Pike in the 2009 Fox television film Virtuality , originally intended as a pilot . Fox television Fox Broadcasting Company New Amsterdam New Amsterdam (TV series) Virtuality Virtuality (TV series) | <Fox_Broadcasting_Company> line 0: The Fox Broadcasting Company -LRB- often shortened to Fox and stylized as FOX -RRB- is an American English language commercial broadcast television network that is owned by the Fox Entertainment Group subsidiary of 21st Century Fox .\tEnglish\tEnglish language\tFox Entertainment Group\tFox Entertainment Group\t21st Century Fox\t21st Century Fox\tAmerican\tUnited States\tEnglish language\tEnglish language\tcommercial\tcommercial broadcasting\tbroadcast\tterrestrial television\ttelevision network\ttelevision network
Thought: I have found evidence. I need to determine the verdict label the claim.
Action: Verdict_Predictor
Action Input: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. | <Nikolaj_Coster-Waldau> line 7 ; <Fox_Broadcasting_Company> line 0
Observation: The evidence supports the claim because it provides clear and direct information that Nikolaj Coster-Waldau played the role of Detective John Amsterdam in the Fox television series New Amsterdam in 2008 and also appeared as Frank Pike in the 2009 Fox television film Virtuality. This information directly establishes his work with the Fox Broadcasting Company, confirming the claim's validity.
The answer is supports
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is supported

Claim: Stranger Things is set in Bloomington, Indiana.
Passage Names: Stranger_Things
Thought: I should find evidence sentences from passages. I should input in the format: claim | names of passagess
Action: Evidence_Seeker
Action Input: Stranger Things is set in Bloomington, Indiana. | Stranger_Things
Observation: <Stranger_Things> line 5: Set in the fictional town of Hawkins , Indiana in the 1980s , the first season focuses on the investigation into the disappearance of a young boy by his friends , older brother and traumatized mother and the local police chief , amid supernatural events occurring around the town including the appearance of a psychokinetic girl who helps the missing boy 's friends in their own search .\tpsychokinetic\tpsychokinesis
Thought: I have found evidence. I need to determine the verdict label the claim.
Action: Verdict_Predictor
Action Input: Stranger Things is set in Bloomington, Indiana. | <Stranger_Things> line 5
Observation: The evidence refutes the claim because it specifies that 'Stranger Things' is set in the fictional town of Hawkins, Indiana, which implies it is not set in Bloomington, Indiana.
The answer is refutes
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is refuted

Claim: System of a Down briefly disbanded in limbo.
Passage Names: Title_in_Limb
Thought: I should find evidence sentences from passages. I should input in the format: claim | names of passagess
Action: Evidence_Seeker
Action Input: System of a Down briefly disbanded in limbo. | Title_in_Limbo
Observation: no evidence
Thought: I can give an answer now
Final Answer: there is not enough information
"""

EXAMPLES_GOLD_EVIDENCE = """
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Evidence: <Nikolaj_Coster-Waldau> line 7 ; <Fox_Broadcasting_Company> line 0
Thought: I have known evidence. I need to determine the verdict label the claim.
Action: Verdict_Predictor
Action Input: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. | <Nikolaj_Coster-Waldau> line 7 ; <Fox_Broadcasting_Company> line 0
Observation: The evidence supports the claim because it provides clear and direct information that Nikolaj Coster-Waldau played the role of Detective John Amsterdam in the Fox television series New Amsterdam in 2008 and also appeared as Frank Pike in the 2009 Fox television film Virtuality. This information directly establishes his work with the Fox Broadcasting Company, confirming the claim's validity.
The answer is supports
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is supported

Claim: Stranger Things is set in Bloomington, Indiana.
Evidence: <Stranger_Things> line 5
Thought: I have found evidence. I need to determine the verdict label the claim.
Action: Verdict_Predictor
Action Input: Stranger Things is set in Bloomington, Indiana. | <Stranger_Things> line 5
Observation: The evidence refutes the claim because it specifies that 'Stranger Things' is set in the fictional town of Hawkins, Indiana, which implies it is not set in Bloomington, Indiana.
The answer is refutes
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is refuted

Claim: System of a Down briefly disbanded in limbo.
Evidence: <Title_in_Limb> line 3
Thought: I have found evidence. I need to determine the verdict label the claim.
Action: Verdict_Predictor
Action Input: System of a Down briefly disbanded in limbo. | <Title_in_Limb> line 3
Observation: The evidence does not either refute or support the claim
there is not enough information
Thought: I can give an answer now
Final Answer: there is not enough information
"""

EXAMPLES_VERIFICATION = f"""
Passage Names: Nikolaj_Coster-Waldau ; Fox_Broadcasting_Company
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought: I need to verify the claim based on retrieved passages.
Action: Evidence_Seeker
Action Input: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. | Nikolaj_Coster-Waldau ; Fox_Broadcasting_Company
Observation: <Nikolaj_Coster-Waldau> line 7: He then played Detective John Amsterdam in the short-lived Fox television series New Amsterdam -LRB- 2008 -RRB- , as well as appearing as Frank Pike in the 2009 Fox television film Virtuality , originally intended as a pilot . Fox television Fox Broadcasting Company New Amsterdam New Amsterdam (TV series) Virtuality Virtuality (TV series)
suggested answer: supports
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is supported

Passage Names: Adrienne_Bailon
Claim: Adrienne Bailon is an accountant.
Thought: I need to verify the claim based on retrieved passages.
Action: Evidence_Seeker
Action Input: Adrienne Bailon is an accountant. | Adrienne_Bailon
Observation: <Adrienne_Bailon> line 0: Adrienne Eliza Houghton -LRB- née Bailon ; born October 24 , 1983 -RRB- is an American singer-songwriter , recording artist , actress , dancer and television personality . née Name at birth singer-songwriter singer-songwriter
suggested answer: refutes
Thought: I have found enough evidences. I can give an answer based on the evidence
Final Answer: the claim is refuted
"""

EXAMPLES_ENTAILMENT = f"""
Passage Names: Nikolaj_Coster-Waldau line 7 ; Fox_Broadcasting_Company line 0
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought: I need to verify the claim based on given evidence.
Action: Entailment_Classifier
Action Input: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. | Nikolaj_Coster-Waldau line 7 ; Fox_Broadcasting_Company line 0
Observation: supports

Passage Names: Adrienne_Bailon line 0
Claim: Adrienne Bailon is an accountant.
Thought: I need to verify the claim based on given evidence.
Action: Entailment_Classifier
Action Input: Adrienne Bailon is an accountant. | Adrienne_Bailon line 0
Observation: refutes
"""

# Query Prediction
query_prediction_prefix = '''
You and your partners are on a mission to fact-check a claim. Imagine yourself as an expert in searching information on Wikipedia.
Your task is to generate a search query that would likely be the title of a Wikipedia page related to the given claim. Wikipedia page titles are typically concise and descriptive, reflecting the subject of the article.
Please predict titles for potentially relevant articles. You may use your expertise to find titles for all relevant passages.

To illustrate the task, here are some examples:'''

query_template = """
Claim: {claim}
Query: {wiki_name}"""

query_prompt = PromptTemplate(
    input_variables=["claim","wiki_name"],
    template=query_template,
)

query_example_f = os.path.join(prompt_dir, 'query_example.jsonl')
query_example = []
with open(query_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        # item['wiki_name'] = item['wiki_name']
        query_example.append(item)

query_fs_prompt = FewShotPromptTemplate(
    examples=query_example,
    example_prompt=query_prompt,
    prefix=query_prediction_prefix,
    suffix="\nNow, let's return to your task. You are given the following claim, please predict possible search queries for it.\n\nClaim: {claim}\nQuery:",
    input_variables=["claim"],
    example_separator="\n",
)

# Evidence Selection
evidence_selection_prefix = '''
Your mission is to verify a claim's factual accuracy. As experts in reading comprehension, you'll receive a claim and a passage. The passage includes a title, and lines are indexed.
You should first read the claim and the passage carefully. Make sure you understand what information you are looking for. Then select lines that either support or refute the claim. A line supports the claim if the information in the line provides evidence, data, or reasoning that strengthens or affirms the validity of the claim. A line refutes the claim if the line contains any evidence or reasoning that contradicts or disproves any information in the claim. Otherwise, we consider a line does not provide information either supporting or refuting the claim. Provide line indices if the passage contains pertinent information. Report 'not enough info' if the passage lacks lines either supporting or refuting the claim.
You should be very cautious in your selection and judgement and avoid overstatement. Base decisions only on provided information without implying additional details.

To illustrate the task, here are some examples:
'''

evidence_template = """
Claim: {claim}
Passage:
{wiki_passage}
Analysis: {analysis}"""

evidence_prompt = PromptTemplate(
    input_variables=["claim", "wiki_passage", "analysis"],
    template=evidence_template,
)

evidence_example_f = os.path.join(prompt_dir, 'evidence_selection.jsonl')
evidence_example = []
with open(evidence_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        # instance = {'claim': item['claim'], 'wiki_passage': item['wiki_passage'], 'evidence': item['evidence']}
        evidence_example.append(item)

evidence_fs_prompt = FewShotPromptTemplate(
    examples=evidence_example,
    example_prompt=evidence_prompt,
    prefix=evidence_selection_prefix,
    suffix="\nNow, let's focus on your task. You are given a claim and a passage. Please read the passage carefully and find lines that contain information supporting or refuting the claim.\n\nClaim: {claim}\nPassage:\n{passage}\nLet's think step by step.\nAnalysis:",
    input_variables=["claim", "passage"],
    example_separator="\n",
)

# Verdict Prediction
verdict_prediction_prefix = '''
Your mission is to verify the factual accuracy of a claim using provided evidence. Your partners have collected evidence, and your expertise lies in assessing the claim's factualness based on this evidence.
You are required to determine whether the claim is supported, refuted, or lacks sufficient information based on the provided evidence. The evidence supports the claim if it confirms all statements and details in the claim. The evidence refutes the claim if it contradicts or disproves any statement in the claim. 'Not enough info' applies when the evidence lacks sufficient data, details, or reasoning to support or refute the claim. Even if the evidence supports part of the claim, it should be considered "not enough info" if there is any detail or statement in the claim that cannot be confirmed by the evidence.
Please exercise caution in making judgments and avoid overstatement. Base decisions solely on the provided information without implying additional details.

Here are examples to illustrate the task:
'''

entailment_template = """
Claim: {claim}
Evidence: {evidence}
Conclusion: {analysis}"""

entailment_prompt = PromptTemplate(
    input_variables=["claim", "evidence","analysis"],
    template=entailment_template,
)

entailment_example_f = os.path.join(prompt_dir,'verdict_prediction.jsonl')
entailment_example = []
with open(entailment_example_f, "r", encoding="utf-8") as reader:
    for item in jsonlines.Reader(reader):
        entailment_example.append(item)

entailment_fs_prompt = FewShotPromptTemplate(
    examples=entailment_example,
    example_prompt=entailment_prompt,
    prefix=verdict_prediction_prefix,
    suffix="\nNow, let's return to your task. You are given the following claim and evidence.\n\nClaim: {claim}\nEvidence: {evidence}\nConclusion:",
    input_variables=["claim", "evidence"],
    example_separator="\n",
)

# Llama
query_prediction_prefix_llama = '''
You an expert in searching information on Wikipedia.
Your task is to generate a search query that would likely be the title of a Wikipedia page related to the given claim. Wikipedia page titles are typically concise and descriptive, reflecting the subject of the article.
Please predict titles for potentially relevant articles. You may use your expertise to find titles for all relevant passages.
You should only provide the predicted titles without any other word. Use ; to separate different titles

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
'''

query_fs_prompt_llama = FewShotPromptTemplate(
    examples=query_example,
    example_prompt=query_prompt,
    prefix='[INST] <<SYS>>'+query_prediction_prefix_llama,
    suffix="<</SYS>>\nNow, let's return to your task. You are given the following claim, please predict possible search queries for it.\nClaim: {claim}\n[/INST]\nQuery:",
    input_variables=["claim"],
    example_separator="\n",
)

# Evidence Selection
evidence_selection_prefix_llama = '''
You are an expert in reading comprehension, you'll receive a claim and a passage. The passage includes a title, and lines are indexed.
You should first read the claim and the passage carefully. Make sure you understand what information you are looking for. Then select lines that either support or refute the claim. 

A line supports the claim if the information in the line provides evidence, data, or reasoning that strengthens or affirms the validity of the claim. 
A line refutes the claim if the line contains any evidence or reasoning that contradicts or disproves any information in the claim. 
Otherwise, we consider a line does not provide information either supporting or refuting the claim. 

Provide line indices if the passage contains pertinent information. Report 'not enough info' if the passage lacks lines either supporting or refuting the claim.
Please select evidence carefully and do not omit any useful information.

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
'''

evidence_fs_prompt_llama = FewShotPromptTemplate(
    examples=evidence_example,
    example_prompt=evidence_prompt,
    prefix='[INST] <<SYS>>'+evidence_selection_prefix_llama,
    suffix="<</SYS>>\nNow, let's focus on your task. You are given a claim and a passage. Please find lines that contain information that can support or refute the claim. Do not omit any useful information.\nClaim: {claim}\nPassage:\n{passage}\n[/INST]\nAnalysis:",
    input_variables=["claim", "passage"],
    example_separator="\n",
)

# Verdict Prediction
verdict_prediction_prefix_llama = '''
You are an expert in verifying the factual accuracy of a claim using provided evidence. 
You are required to determine the verdict label of the claim based on the provided evidence. Remember that the label must be one of [SUPPORTED, REFUTED, NOT ENOUGH INFO]. Other labels are not valid. 
The evidence supports the claim if it confirms statements and details in the claim. The evidence refutes the claim if it contradicts or disproves any statement in the claim. 'Not enough info' applies when the evidence lacks sufficient data, details, or reasoning to support or refute the claim. Even if the evidence supports part of the claim, it should be considered "not enough info" if there is any detail or statement in the claim that cannot be confirmed by the evidence.

To illustrate the task, below are some examples. Please make sure that your outputs follow the example format.
'''

entailment_fs_prompt_llama = FewShotPromptTemplate(
    examples=entailment_example,
    example_prompt=entailment_prompt,
    prefix='[INST] <<SYS>>'+verdict_prediction_prefix_llama,
    suffix="<</SYS>>\nNow, let's return to your task. Analyze the factualness of the claim based on the evidence.\nClaim: {claim}\nEvidence: {evidence}\n[/INST]\nConclusion:",
    input_variables=["claim", "evidence"],
    example_separator="\n",
)

