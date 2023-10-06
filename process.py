import os

import hkkang_utils.file as file_utils
import torch
import tqdm

# Read corpus
corpus_path = "/root/dpr_scale/data/wiki/enwiki-2017-passage-corpus.tsv"
corpus = file_utils.read_csv_file(corpus_path, delimiter="\t")
id_mapping = {datum["id"]: idx for idx, datum in enumerate(corpus)}

path = "/root/dpr_scale/ckpts/citadel/hotpotqa_merged/expert/"
filenames = file_utils.get_files_in_directory(path)
for filename in tqdm.tqdm(filenames):
    full_file_path = os.path.join(path, filename)
    data = file_utils.read_pickle_file(full_file_path)
    ids = data[0]
    correct_ids = []
    for id in ids:
        # Convert id
        correct_ids.append(id_mapping[f"{id}_0"])
    correct_ids_tensor = torch.tensor(correct_ids)
    data[0] = correct_ids_tensor
    # Write new data
    file_utils.write_pickle_file((data[0], data[1], data[3]), full_file_path)
print("Done!")