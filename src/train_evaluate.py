# src/train_evaluate.py

import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.model import DiseaseGNN

def train_model(graph, num_epochs=60):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    graph = graph.to(device)

    input_dim = graph.x.shape[1]
    model = DiseaseGNN(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    num_neg = (graph.edge_attr == 0).sum().item()
    num_pos = (graph.edge_attr == 1).sum().item()
    pos_weight = num_neg / max(num_pos, 1)
    weight = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    print(f"  Neg: {num_neg}  Pos: {num_pos}  Pos-weight: {pos_weight:.1f}x")
    print(f"\n  {'Epoch':>6}  {'Loss':>8}  {'Acc':>7}  {'CID-F1':>8}")
    print("  " + "-"*36)

    loss_history = []
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        out  = model(graph.x, graph.edge_index)
        loss = criterion(out, graph.edge_attr)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            preds = out.argmax(dim=1)
            acc = (preds == graph.edge_attr).float().mean().item()
            tp = ((preds==1)&(graph.edge_attr==1)).sum().item()
            fp = ((preds==1)&(graph.edge_attr==0)).sum().item()
            fn = ((preds==0)&(graph.edge_attr==1)).sum().item()
            f1 = 2*tp / max(2*tp+fp+fn, 1e-8)
            print(f"  {epoch:>6}  {loss.item():>8.4f}  {acc:>7.4f}  {f1:>8.4f}")

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(loss_history, color='#065A82', linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig("outputs/loss_curve.png", dpi=150)
    plt.close()
    print("  Loss curve saved -> outputs/loss_curve.png")
    return model

def evaluate_model(model, graph, split_name="Evaluation"):
    device = next(model.parameters()).device
    graph  = graph.to(device)
    model.eval()
    with torch.no_grad():
        out    = model(graph.x, graph.edge_index)
        preds  = out.argmax(dim=1).cpu().numpy()
        labels = graph.edge_attr.cpu().numpy()

    print(f"\n{'='*50}")
    print(f"  {split_name}")
    print(f"{'='*50}")
    print(classification_report(labels, preds,
          target_names=["No Relation", "CID Relation"], digits=4))

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Rel","CID"]); ax.set_yticklabels(["No Rel","CID"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {split_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    fname = f"outputs/confusion_{split_name.lower().replace(' ','_')}.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Confusion matrix saved -> {fname}")