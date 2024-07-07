import json
import argparse
import os
import requests
from sklearn import metrics
from sklearn.metrics import *
from sentence_transformers import SentenceTransformer, util
import torch

similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
THRESHOLD = 0.8


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="path of example file",
    )    
    args = parser.parse_args()

    return args

def evidence_similarity(sents, ref_sents):
    embedding_1= similarity_model.encode(sents, convert_to_tensor=True).to('cuda')
    embedding_2 = similarity_model.encode(ref_sents, convert_to_tensor=True).to('cuda')
    similarities = util.pytorch_cos_sim(embedding_1, embedding_2).tolist()

    return similarities


def evidence_macro_precision(similarities, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0
    prediction_sim = similarities if max_evidence is None else similarities[:max_evidence]
    for prediction in prediction_sim: # a ref sentence v.s. all retrieved sentences
        # print(prediction)
        if max(prediction) > THRESHOLD: # if there exist at least one retrieved sentence over the threshold
            this_precision += 1.0
        this_precision_hits += 1.0
    return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

def evidence_macro_recall(sim, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    # If there's no evidence to predict, return 1
    if len(sim[0]) == 0 or len(sim) == 0:
        return 1.0, 1.0
    this_recall = 0.0
    this_recall_hits = 0.0
    predicted_sim = sim if max_evidence is None else sim[:max_evidence]
    transposed_predicted_sim = [list(i) for i in zip(*predicted_sim)]

    for ref_sent in transposed_predicted_sim:
        if max(ref_sent) > THRESHOLD: # If a ref sentence is not included in the prediction
            this_recall += 1.0
        this_recall_hits += 1.0
    return (this_recall / this_recall_hits) if this_recall_hits > 0 else 1.0, 1.0
    

def evidence_score(sents_list, ref_sents_list, max_evidence=5):
    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0

    for sents, ref_sents in zip(sents_list, ref_sents_list):
        if len(ref_sents) < 1: # not supported claim
            macro_precision += 0.0
            macro_precision_hits += 0.0
            macro_recall += 0.0
            macro_recall_hits += 0.0
            continue
        
        sim = evidence_similarity(sents, ref_sents)
        macro_prec = evidence_macro_precision(sim, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(sim, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)
    return pr, rec, f1



if __name__=='__main__':
    args = parse_args()
    assert args.file_path, 'specify the evaluation file'
    
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bingcheck")
    all_files = [f for f in os.listdir(file_path) if f.endswith('.json')]
    file_name_dict = {f_name.split('.json')[0].split('_')[-1]: f_name for f_name in all_files}

    labels = []
    preds = []
    golden_knowledge = []
    pred_knowledge = []

    with open(args.file_path) as f:
        data = json.load(f)
        for item_id in list(data.keys()):
            item = data[item_id]
            if item_id not in file_name_dict:
                print(item_id)
                continue
            with open(os.path.join(file_path,file_name_dict[item_id])) as ref:
                ref_data = json.load(ref)
                subconclusions = list(set(list(ref_data[item_id]["fact-checking"]['subconclusions'].values())))
                if 'refuted' in subconclusions:
                    label = 'refute'
                elif 'partially supported' in subconclusions:
                    label = 'partially support'
                elif 'not supported' in subconclusions:
                    if len(subconclusions) == 1: # all subclaims are not supported
                        label = 'not support'
                    else:
                        label = 'partially support'
                else:
                    label = 'support'
                
                knowledge_list = ref_data[item_id]["fact-checking"]['evidence']
                flatten_knowledge_list = []
                for claim in knowledge_list:
                    flatten_knowledge_list += knowledge_list[claim]
                flatten_knowledge_list = list(set(flatten_knowledge_list))
                golden_knowledge.append(flatten_knowledge_list)
                

            if 'subconclusions' in item:
                subconclusions = list(set(list(item['subconclusions'].values())))
                if 'refuted' in subconclusions:
                    pred = 'refute'
                elif 'partially supported' in subconclusions:
                    pred = 'partially support'
                elif 'not supported' in subconclusions:
                    if len(subconclusions) == 1: # all subclaims are not supported
                        pred = 'not support'
                    else:
                        pred = 'partially support'
                else:
                    pred = 'support'

                knowledge_list = item['evidence']
                flatten_knowledge_list = []
                for claim in knowledge_list:
                    flatten_knowledge_list += knowledge_list[claim]
                flatten_knowledge_list = list(set(flatten_knowledge_list))
                pred_knowledge.append(flatten_knowledge_list)
                
            else:
                if 'refute' in item['conclusion'].replace('_',' ').lower():
                    pred = 'refute'
                elif 'partially support' in item['conclusion'].replace('_',' ').lower():
                    pred = 'partially support'
                elif 'not support' in item['conclusion'].replace('_',' ').lower():
                    pred = 'not support'
                else:
                    pred = 'support'
            
            preds.append(pred)
            labels.append(label)


    print(f'TOTAL: {len(preds)}')
    print("Classification:\n")
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    print(json.dumps({"accuracy": accuracy}))

    precision, recall, f1 = evidence_score(pred_knowledge, golden_knowledge, max_evidence=None)
    results = {"f1": f1, "precision": precision, "recall": recall}
    print(json.dumps(results, indent=2))



