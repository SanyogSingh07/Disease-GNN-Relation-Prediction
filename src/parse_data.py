# src/parse_data.py

def parse_pubtator(filepath):
    documents = []
    current_doc = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_doc:
                    documents.append(current_doc)
                    current_doc = None
                continue
            if '|t|' in line:
                parts = line.split('|t|', 1)
                current_doc = {'pmid': parts[0], 'title': parts[1],
                               'abstract': '', 'entities': [], 'relations': []}
            elif '|a|' in line and current_doc:
                current_doc['abstract'] = line.split('|a|', 1)[1]
            elif '\tCID\t' in line and current_doc:
                parts = line.split('\t')
                if len(parts) == 4:
                    current_doc['relations'].append((parts[2].strip(), parts[3].strip()))
            elif current_doc and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 6:
                    try:
                        current_doc['entities'].append({
                            'start': int(parts[1]), 'end': int(parts[2]),
                            'text': parts[3], 'type': parts[4],
                            'mesh_id': parts[5].strip()
                        })
                    except ValueError:
                        pass

    if current_doc:
        documents.append(current_doc)
    return documents


if __name__ == "__main__":
    import os
    docs = parse_pubtator(os.path.join("data", "CDR_TrainingSet.PubTator.txt"))
    print(f"Loaded {len(docs)} documents")
    print("Title:", docs[0]['title'][:80])
    print("Relations:", docs[0]['relations'])
    chemicals = sum(1 for d in docs for e in d['entities'] if e['type'] == 'Chemical')
    diseases  = sum(1 for d in docs for e in d['entities'] if e['type'] == 'Disease')
    print(f"Total chemicals: {chemicals}, diseases: {diseases}")