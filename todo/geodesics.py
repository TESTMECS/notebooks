import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = bert(**inputs).last_hidden_state
    return output[0, 1]  # embedding of first word token


# Step 1: Endpoints
a = get_embedding("The sun is bright").detach()
b = get_embedding("The moon is dark").detach()

# Step 2: Interpolated path (T steps)
T = 20
dim = a.shape[0]
with torch.no_grad():
    linear_path = torch.stack([a + (b - a) * t / (T - 1) for t in range(T)])
intermediate = nn.Parameter(linear_path[1:-1].clone())  # make [x1, ..., xT-1] trainable


# Step 3: Define energy functional
def energy(path):
    diffs = path[1:] - path[:-1]
    return (diffs**2).sum(dim=1).mean()  # average squared velocity = path energy


# Step 4: Optimization loop
optimizer = torch.optim.Adam([intermediate], lr=1e-2)

for step in range(500):
    full_path = torch.cat([a.unsqueeze(0), intermediate, b.unsqueeze(0)], dim=0)
    loss = energy(full_path)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}, Energy: {loss.item():.4f}")

# Optional: Visualize cosine similarity across the path
with torch.no_grad():
    final_path = torch.cat([a.unsqueeze(0), intermediate, b.unsqueeze(0)], dim=0)
    sims = torch.nn.functional.cosine_similarity(final_path[:-1], final_path[1:], dim=1)
    plt.plot(sims.numpy())
    plt.title("Cosine similarity between steps along geodesic")
    plt.ylabel("cos(θ)")
    plt.xlabel("step")
    plt.show()
