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
        "--fever_path",
        type=str,
        default=None,
        help="path to fever dataset",
    ) 
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = parse_args()
    conn = sqlite3.connect('wiki_pages.db') 
    c = conn.cursor()
    c.execute('''
            CREATE TABLE IF NOT EXISTS wikipages
            ([id] TEXT PRIMARY KEY, [norm_id] TEXT, [text] TEXT, [lines] TEXT)
            ''')
    passage_path = os.path.join(args.fever_path,'wiki-pages')
    file_list = os.listdir(passage_path)
    for file_name in file_list:
        with open(os.path.join(passage_path, file_name)) as wiki_pages:  
            for item in tqdm.tqdm(jsonlines.Reader(wiki_pages)):
                print(item)
                c.execute("INSERT INTO wikipages VALUES (?, ?, ?, ?)", (item['id'], normalize_name(item['id']), item['text'], item['lines']))
    c.execute("Create Index myIndex On wikipages(norm_id);")
    conn.commit()