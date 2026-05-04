# main.py

import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))

from src.parse_data     import parse_pubtator
from src.build_graph    import build_graph
from src.train_evaluate import train_model, evaluate_model

print("="*55)
print("  GNN for Chemical-Disease Relation Prediction")
print("="*55)

print("\n[1/4] Loading datasets...")
train_docs = parse_pubtator("data/CDR_TrainingSet.PubTator.txt")
dev_docs   = parse_pubtator("data/CDR_DevelopmentSet.PubTator.txt")
test_docs  = parse_pubtator("data/CDR_TestSet.PubTator.txt")
print(f"  Train: {len(train_docs)} | Dev: {len(dev_docs)} | Test: {len(test_docs)} docs")

print("\n[2/4] Building graphs...")
train_graph, entity_map, vectorizer = build_graph(train_docs, fit_vectorizer=True)
dev_graph,   _, _ = build_graph(dev_docs,  vectorizer=vectorizer, fit_vectorizer=False)
test_graph,  _, _ = build_graph(test_docs, vectorizer=vectorizer, fit_vectorizer=False)

print("\n[3/4] Training GNN...")
model = train_model(train_graph, num_epochs=60)

print("\n[4/4] Evaluating...")
evaluate_model(model, train_graph, "Training Set")
evaluate_model(model, dev_graph,   "Development Set")
evaluate_model(model, test_graph,  "Test Set")

os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/gnn_model.pt")
print("\nDone! Check the outputs/ folder for plots and results.")