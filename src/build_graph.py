# src/build_graph.py

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data

def build_graph(documents, vectorizer=None, fit_vectorizer=True):
    entity_to_idx  = {}
    entity_context = {}

    for doc in documents:
        full_text = doc['title'] + " " + doc['abstract']
        for ent in doc['entities']:
            mid = ent['mesh_id']
            if mid in ('-1', ''):
                continue
            if mid not in entity_to_idx:
                entity_to_idx[mid] = len(entity_to_idx)
                entity_context[mid] = []
            entity_context[mid].append(full_text)

    num_nodes = len(entity_to_idx)
    print(f"  Nodes: {num_nodes}")

    node_docs = [" ".join(entity_context[mid]) for mid in entity_to_idx]
    if fit_vectorizer or vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
        X = vectorizer.fit_transform(node_docs).toarray()
    else:
        X = vectorizer.transform(node_docs).toarray()

    node_features = torch.tensor(X, dtype=torch.float)

    edge_src, edge_dst, edge_labels = [], [], []

    for doc in documents:
        doc_cid = set()
        for chem_id, dis_id in doc['relations']:
            doc_cid.add((chem_id, dis_id))
            doc_cid.add((dis_id, chem_id))

        valid = [e for e in doc['entities']
                 if e['mesh_id'] not in ('-1', '') and e['mesh_id'] in entity_to_idx]

        seen = set()
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                ei, ej = valid[i]['mesh_id'], valid[j]['mesh_id']
                pair = tuple(sorted([ei, ej]))
                if pair in seen:
                    continue
                seen.add(pair)
                ii, ij = entity_to_idx[ei], entity_to_idx[ej]
                label = 1 if (ei, ej) in doc_cid else 0
                edge_src.extend([ii, ij])
                edge_dst.extend([ij, ii])
                edge_labels.extend([label, label])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_labels, dtype=torch.long)

    pos = int(edge_attr.sum().item()) // 2
    print(f"  Edges: {edge_index.shape[1] // 2}  |  Positive CID pairs: {pos}")

    graph = Data(x=node_features, edge_index=edge_index,
                 edge_attr=edge_attr, num_nodes=num_nodes)
    return graph, entity_to_idx, vectorizer