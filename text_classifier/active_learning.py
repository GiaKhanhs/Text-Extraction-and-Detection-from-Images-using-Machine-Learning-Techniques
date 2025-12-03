import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# Split dataset into L and U
def split_dataset_indices(total_samples, init_L=500):
    indices = list(range(total_samples))
    L_indices, U_indices = train_test_split(indices, train_size=init_L, random_state=42)
    return L_indices, U_indices

# Compute entropy
def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean(dim=-1)  # mean over time steps
    return entropy

# Select top-K high-entropy samples
def select_topk_entropy(model, dataset_full, U_indices, K, device):
    scores = []
    model.eval()
    with torch.no_grad():
        for idx in tqdm(U_indices, desc="Evaluating entropy"):
            sample = dataset_full[idx]
            img = sample['img'].unsqueeze(0).to(device)
            tgt_input = torch.LongTensor(sample['word']).unsqueeze(0).to(device)

            logits = model(img, tgt_input)
            entropy = compute_entropy(logits)
            scores.append((idx, entropy.item()))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    selected = [x[0] for x in sorted_scores[:K]]
    return selected

# Update L and U
def update_L_U(L_indices, U_indices, selected_indices):
    L_indices += selected_indices
    U_indices = list(set(U_indices) - set(selected_indices))
    return L_indices, U_indices

# Placeholder train and eval (replace with your own)
def train_model_on_L(trainer, dataset_L):
    trainer.train_loader = torch.utils.data.DataLoader(dataset_L, batch_size=32, shuffle=True, collate_fn=trainer.collate_fn)
    trainer.train()

def evaluate_on_val(trainer):
    acc, cer = trainer.precision()
    return acc, cer

# Main Active Learning Loop
def run_active_learning(trainer, dataset_full, device, init_L=500, K=100, max_rounds=10):
    L_indices, U_indices = split_dataset_indices(len(dataset_full), init_L=init_L)

    for round in range(max_rounds):
        print(f"\n===== ROUND {round + 1} =====")
        dataset_L = Subset(dataset_full, L_indices)

        # Train
        train_model_on_L(trainer, dataset_L)

        # Evaluate
        acc, cer = evaluate_on_val(trainer)
        print(f"Validation - Accuracy: {acc:.4f}, CER: {cer:.4f}")

        if len(U_indices) == 0:
            print("Unlabeled pool exhausted.")
            break

        # Select samples to add to L
        selected = select_topk_entropy(trainer.model, dataset_full, U_indices, K, device)

        # Update L and U
        L_indices, U_indices = update_L_U(L_indices, U_indices, selected)