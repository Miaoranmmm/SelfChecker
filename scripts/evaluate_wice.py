import json
import re
import os
import argparse
from collections import Counter
from sklearn import metrics
from sklearn.metrics import *
import jsonlines


final_results = []

def evidence_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"] != "not_supported":
        evi_list = instance["evidence"]
        all_evi = [item for sublist in evi_list for item in sublist]

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

def evidence_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"] != "not_supported":
        
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
           return 1.0, 1.0

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]
        evidence = instance['evidence']
        
        for evi in evidence:
            if all([item in predicted_evidence for item in evi]):
                    # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0

def evidence_score(predictions, max_evidence=5):
    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0
    for idx,instance in enumerate(predictions):
        macro_prec = evidence_macro_precision(instance, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]
    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)
    return pr, rec, f1



def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="path of example file",
    )    
    parser.add_argument(
        "--baseline",
        action='store_true',
        help="always entailed",
    )
    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = parse_args()
    assert args.file_path, 'specify the evaluation file'
    
    with open(args.file_path) as f:
        pred_data = json.load(f)

    preds = []
    labels = []

    with jsonlines.open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wice/data/entailment_retrieval/claim/test.jsonl")) as test_ref:
        for item in test_ref:
            item_id = item['meta']['id']
            if item_id not in pred_data:
                print(item_id)
                continue
            pred_item = pred_data[item_id]
            conclusion = pred_item['conclusion'].lower()
            
            if 'not support' in conclusion or 'partially support' in conclusion:
                pred_label = 0
            elif 'support' in conclusion:
                pred_label = 1
            else:
                print(item_id, pred_item)
                pred_label = 0
            label = 1 if item['label'] == 'supported' else 0
            
            preds.append(pred_label)
            labels.append(label)

            # evidence retrieval eval
            if item['label'] == 'not_supported':
                continue
            gold_evi = item['supporting_sentences']
            try:
                subclaim_results = pred_item['subclaim_results']
            except:
                continue
            evidence = []
            for subclaim_result in subclaim_results:
                evidence += subclaim_result['evidence']
            instance = {
                "label": item['label'],
                "predicted_evidence": list(set(evidence)),
                "evidence": gold_evi # list of list
            }
            final_results.append(instance)
   
            
            
            
    print(f'TOTAL: {len(preds)}')
    print("Entailment Classification:\n")
    prfs = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='binary')
    print({"f1": prfs[2], "precision": prfs[0], "recall": prfs[1]})
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    print(json.dumps({"accuracy": accuracy}))

    print("\nEvidence Retrieval:\n")
    precision, recall, f1 = evidence_score(final_results)
    results = {"f1": f1, "precision": precision, "recall": recall}
    print(json.dumps(results, indent=2))
    
