import json
import jsonlines
import re
import argparse
from fever_scorer.src.fever.scorer import fever_score
from sklearn import metrics
import matplotlib.pyplot as plt

final_results = []
regex = r'\<(.*)\>'

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

if __name__=='__main__':
    args = parse_args()
    assert args.file_path, 'specify the evaluation file'

    labels_truth = []
    preds = []
    with open(args.file_path) as f:
        data = json.load(f)
        num = 0
        for idx in data:
            item = data[idx]
            conclusion = item['conclusion'].lower()
            if 'support' in conclusion or 'refute' in conclusion or 'not enough info' in conclusion:
                if 'not enough info' in conclusion:
                    pred_label = 'NOT ENOUGH INFO'
                elif 'support' in conclusion:
                    pred_label = 'SUPPORTS'
                elif 'refute' in conclusion:
                    pred_label = 'REFUTES'
            elif 'not enough' in conclusion or 'cannot be verified' in conclusion:
                pred_label = 'NOT ENOUGH INFO'
            elif 'yes,' in conclusion or 'true' in conclusion:
                pred_label = 'SUPPORTS'
            elif 'no,' in conclusion or 'false' in conclusion or 'not' in conclusion:
                pred_label = 'REFUTES'
            elif conclusion != item['claim'].lower():
                pred_label = 'REFUTES'
            elif conclusion == item['claim'].lower():
                pred_label = 'SUPPORTS'
            
            label = item['label'].upper()
            labels_truth.append(label)
            preds.append(pred_label)
            
            predicted_evidence = []
            evidence = []
            if pred_label != 'NOT ENOUGH INFO':
                if 'evidence' in item:
                    for piece in item['evidence'].split(' | '):
                        try:
                            predicted_evidence.append([re.match(regex, piece).group(1), int(piece.split(' line ')[1].split(':')[0].strip())])
                        except:
                            pass

            if label != 'NOT ENOUGH INFO':
                if 'gold_evi' in item:
                    for piece in item['gold_evi'].split(' ; '):
                    # for piece in ref[int(idx)]:
                        evidence.append([None, None, re.match(regex, piece).group(1), int(piece.split(' line ')[1].strip())])
                    
            evidence = [evidence]
            instance = {
                "label": label,
                "predicted_label": pred_label,
                "predicted_evidence": predicted_evidence,
                "evidence": evidence
            }
            final_results.append(instance)
    print(final_results[:3])
    
    print(f'TOTAL: {len(final_results)}')
            
    strict_score, label_accuracy, precision, recall, f1 = fever_score(final_results)
    results = {"strict_score": strict_score, "label_accuracy": label_accuracy, "f1": f1, "precision": precision, "recall": recall}
    print(json.dumps(results, indent=2))

    cm = metrics.confusion_matrix(labels_truth, preds, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
    cm = [[round(count/sum(row)*100,2) for count in row] for row in cm]
    print('confusion matrix:')
    print(cm)
    