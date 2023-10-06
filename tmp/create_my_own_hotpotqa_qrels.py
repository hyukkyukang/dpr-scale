from typing import *

import hkkang_utils.file as file_utils
import tqdm

data_path = "/root/dpr_scale/data/hotpotqa/dev.json"
corpus_path = "/root/dpr_scale/data/wiki/enwiki-2017-passage-corpus.tsv"


# Read dev data and corpus
data = file_utils.read_json_file(data_path)
corpus = file_utils.read_csv_file(corpus_path, delimiter="\t")
# Create corpus mapping dict

corpus_dict = {d["title"]: idx for idx, d in enumerate(corpus)}

# Find gold doc id mapping for each question
mapping_list: List[Dict[str, str]] = []
for datum in tqdm.tqdm(data):
    qid = datum["id"]
    gold_titles = []
    
    for title, text in datum["supporting_facts"]:
        if title not in gold_titles:
            gold_titles.append(title)
    for gold_title in gold_titles:
        tmp_dict = {"query-id": qid, "corpus-id": corpus_dict[gold_title], "score": 1}
        mapping_list.append(tmp_dict)

    
# Save file
output_path = "/root/dpr_scale/data/hotpotqa/dev_qrels.tsv"
file_utils.write_csv_file(mapping_list, output_path, delimiter="\t")
print("Done!")
