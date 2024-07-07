import os
import jsonlines
import sqlite3
import tqdm
import argparse

def normalize_name(name):
    name = name.strip()
    name = name.replace('-LRB-', '').replace('-RRB-', '').replace('-LSB-', '').replace('-RSB-','')
    name = name.replace('-COLON-', '')
    name = name.replace('_', ' ')
    return name

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--wice_path",
        type=str,
        default=None,
        help="path to wice dataset",
    ) 
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = parse_args()
    conn = sqlite3.connect('wice_passages.db') 
    c = conn.cursor()
    c.execute('''
            CREATE TABLE IF NOT EXISTS passages
            ([id] TEXT PRIMARY KEY, [text] TEXT)
            ''')
    passage_path = os.path.join(args.wice_path, 'data/entailment_retrieval/claim/test.jsonl')
    with open(passage_path) as wice_passages:  
        for item in tqdm.tqdm(jsonlines.Reader(wice_passages)):
            c.execute("INSERT INTO passages VALUES (?, ?)", (item['meta']['id'], '\n'.join(f'{line_id}\t{line}' for line_id, line in enumerate(item['evidence']))))
    conn.commit()