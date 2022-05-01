import os
import json
from tqdm import tqdm


def load_qid2text(path: str):
    """
    Args:
        path (str):
    Returns:
        qid2text (Dict[str, Dict[str, str]]):
    """
    print('loading entity textual description from {}'.format(path))
    qid2text = {}
    with open(path, encoding="utf-8") as f:
        # special
        for line in tqdm(f.readlines()):
            if "######" in line:
                line.replace("######", "### ###")
            items = line.strip().split("###")
            qid, label, desc, cla = items # QID, entity name, entity description, class
            if qid in qid2text:
                continue
            qid2text[qid] = {
                "label": label,
                "description": desc,
                "class": cla,
            }
    print('size of entities in {} is : {}'.format(path, len(qid2text)))
    return qid2text


def url2qid(url):
    if url.startswith('<') and url.endswith('>'):
        url = url[1:-1]
    qid = url.split('/')[-1]
    return qid


def load_raw_data(path: str):
    """
    Args:
        path (str):
    Returns:
        dataset (List[Dict[str, Union[str, List[str]]]]): raw data samples
    """
    print("Loading raw data from {}".format(path))
    dataset = []
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            head, relation, tail, question = line.strip().split('\t')
            head_qid = url2qid(head) # qid
            tail_qid = url2qid(tail) # qid
            relation_names = relation.split('/')[1:] # List[str]
            sample = {
                "head_qid": head_qid,
                "relation_names": relation_names,
                "tail_qid": tail_qid,
                "question": question,
            }
            dataset.append(sample)
    print("Loaded {} raw samples from {}".format(len(dataset), path))
    return dataset


def convert_dataset_format(mode: str = 'train'):
    """
    Args:
        mode (str):
    """
    assert mode in ['train', 'valid', 'test'], f"{mode} is not a valid mode"

    raw_data_file = f"{mode}.txt"
    entity_text_file = f"wikidata_desc_cate_{mode}.txt"
    
    qid2text = load_qid2text(entity_text_file)
    raw_dataset = load_raw_data(raw_data_file)

    # format conversion
    idx = 0
    dataset = []
    for raw_sample in raw_dataset:
        """raw_sample (Dict[str, Union[str, List[str]]]): a raw data sample after simple processing"""
        head_qid = raw_sample['head_qid'] # str, QID
        tail_qid = raw_sample['tail_qid'] # str, QID
        
        realtion_texts = raw_sample['relation_names']
        question = raw_sample['question']
        
        head_texts = qid2text[head_qid] # Dict[str, str], all text fileds for this entity qid
        tail_texts = qid2text[tail_qid]
        
        head = " ".join(head_texts.values())
        relation = " ".join(relation_texts)
        tail = " ".join(tail_texts.values())
        
        sample = {
            "id": idx,
            "kbs": {
                "0": [
                    head,
                    head,
                    [
                        [
                            relation,
                            tail,
                        ]
                    ]
                ]
            },
            "text": [
                question,
            ]
        }
        dataset.append(sample)
        idx += 1
    
    # save dataset to disk
    os.makedirs('sq', exists_ok=True)
    if mode == 'valid': mode = 'dev'
    assert mode in ['train', 'dev', 'test'], f"{mode} is not a valid mode to store json dataset"
    with open(f'sq/{mode}.json', 'w') as f:
        json.dump(dataset, f)
    
    # make src-test.txt, tgt-test.txt
    if mode == 'test':
        f_src = open('sq/src-test.txt', 'w')
        f_tgt = open('sq/tgt-test.txt', 'w')

        for sample in dataset:
            head = sample['kbs']['0'][0]
            relation = sample['kbs']['0'][2][0][0]
            tail = sample['kbs']['0'][2][0][1]
            question = sample['text'][0]
            f_src.write("\t".join([head, relation, tail]) + "\n")
            f_tgt.write(question + '\n')
        
        f_src.close()
        f_tgt.close()


if __name__ == '__main__':
    # make train.json, dev.json and test.json
    for mode in ['train', 'valid', 'test']:
        convert_dataset_format(mode)


    