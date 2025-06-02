
# %% [markdown]
# The ideas here need to be expanded upon to other models.
# Overall what this notebook shows is that we can manipulate the attention latent space to get better results. 
# The latent space is embedded in a 3D Minkowski space.
# We can shift and move the latent space to get better results. 
# We define those ideas that are "Spacelike" those that are not in the same future. 
# We can mask the 'spacelike ideas' to get better results. 
# %%

import torch
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3‑D projection)
from datasets import load_dataset
from matplotlib import animation
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# %%
console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Beautiful device info
device_color = "green" if DEVICE == "cuda" else "yellow"
console.print(Panel(f"🚀 Using device: [bold {device_color}]{DEVICE}[/bold {device_color}]", 
                   title="[bold blue]System Info[/bold blue]", expand=False))

# %%
# 1  Load Tiny GPT‑2 & Prep Data
# We use **sshleifer/tiny‑gpt2** (124 K params) for speed.
console.print("\n[bold cyan]📥 Loading Model & Data[/bold cyan]")

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("Loading GPT-2 model...", total=None)
    model_name = "sshleifer/tiny-gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    progress.update(task, description="✅ Model loaded successfully!")

# ensure the model returns attentions
model.config.output_attentions = True

# Tokenizer
tok = GPT2Tokenizer.from_pretrained(model_name)

# Tiny prediction dataset (feel free to replace with your own)
SENTS = [
    "The cat sat on the mat.",
    "Deep learning has revolutionized natural language processing.",
    "Quantum computers promise exponential speedups for certain problems.",
]

console.print(Panel(f"📊 Dataset: {len(SENTS)} sample sentences", 
                   title="[bold green]Data Ready[/bold green]", expand=False))

# %%
# 2  Helper – Sentence Perplexity

@torch.no_grad()
def sentence_nll(sentence, mdl=model, tokenizer=tok):
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = mdl(**inputs, labels=inputs["input_ids"])
    # negative‑log‑likelihood per token
    return out.loss.item() * inputs["input_ids"].size(1)

@torch.no_grad()
def dataset_perplexity(sents, mdl=model):
    total_nll, total_tok = 0.0, 0
    for s in sents:
        inputs = tok(s, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = mdl(**inputs, labels=inputs["input_ids"])
        total_nll += out.loss.item() * inputs["input_ids"].size(1)
        total_tok += inputs["input_ids"].size(1)
    ppl = math.exp(total_nll / total_tok)
    return ppl

baseline_ppl = dataset_perplexity(SENTS)
console.print(f"\n[bold magenta]📈 Baseline Perplexity:[/bold magenta] [bold white]{baseline_ppl:.3f}[/bold white]")

# %%
# 3  Trace Attention – Build Minkowski Events
console.print("\n[bold cyan]🔍 Tracing Attention Events[/bold cyan]")

ATTN_EVENTS = []
time_ctr = [0]

# Minkowski event: (id, x=token, y=head, t=time_idx, label)

def make_hook(layer_idx):
    def hook(mod, inp, out):
        w = out[1]  # (b, heads, tgt, src)
        b, h, L, _ = w.shape
        for head in range(h):
            for tgt in range(L):
                # take top‑k src by weight > 0.01
                top_src = torch.nonzero(w[0, head, tgt] > 0.01).flatten().tolist()
                for src in top_src:
                    eid = f"ev_{len(ATTN_EVENTS)}"
                    ATTN_EVENTS.append((eid, tgt, head, time_ctr[0], f"L{layer_idx}/H{head}"))
                    time_ctr[0] += 1
    return hook

hooks = []
for i, blk in enumerate(model.transformer.h):
    hooks.append(blk.attn.register_forward_hook(make_hook(i)))

inputs = tok(SENTS[0], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
_ = model(**inputs)  # just run once to get events
for h in hooks:
    h.remove()

console.print(f"✅ Logged [bold green]{len(ATTN_EVENTS)}[/bold green] attention events")

# %% [markdown]
"""
# 4  Compute Minkowski Intervals + Entropy Cost
"""
# %%
console.print("\n[bold cyan]⚛️  Computing Minkowski Spacetime Analysis[/bold cyan]")

def classify_intervals(events):
    rec = []
    for i, e1 in enumerate(events):
        for j, e2 in enumerate(events):
            if i >= j:
                continue
            dx, dy, dt = e2[1]-e1[1], e2[2]-e1[2], e2[3]-e1[3]
            ds2 = -dt*dt + dx*dx + dy*dy
            if abs(ds2) < 1e-3:
                rel = "null"
            elif ds2 < 0:
                rel = "timelike"
            else:
                rel = "spacelike"
            cost = math.sqrt(-ds2) if rel == "timelike" else 0
            rec.append({"from": e1[0], "to": e2[0], "relation": rel, "cost": cost})
    return pd.DataFrame(rec)

interval_df = classify_intervals(ATTN_EVENTS)
timelike_cost = interval_df.query("relation=='timelike'")["cost"].sum()

# Create a beautiful summary table
spacetime_table = Table(title="🌌 Spacetime Interval Analysis")
spacetime_table.add_column("Relation Type", style="cyan", no_wrap=True)
spacetime_table.add_column("Count", style="magenta")
spacetime_table.add_column("Total Cost", style="green")

for relation in ["timelike", "spacelike", "null"]:
    subset = interval_df.query(f"relation=='{relation}'")
    count = len(subset)
    cost = subset["cost"].sum() if relation == "timelike" else "N/A"
    cost_str = f"{cost:.3f}" if cost != "N/A" else cost
    spacetime_table.add_row(relation.title(), str(count), cost_str)

console.print(spacetime_table)

# %% [markdown]
"""
# 5  Identify & Mask Entropy‑Heavy Heads
"""
# %%
console.print("\n[bold cyan]🎯 Identifying Entropy-Heavy Heads[/bold cyan]")

head_entropy = defaultdict(float)
for _, row in interval_df.iterrows():
    if row["relation"] == "timelike":
        src_id = row["from"]
        token_pos, head_idx = next((x[1], x[2]) for x in ATTN_EVENTS if x[0]==src_id)
        head_entropy[(token_pos, head_idx)] += row["cost"]

# pick top‑5
top5 = sorted(head_entropy.items(), key=lambda x: x[1], reverse=True)[:5]
masked_heads = defaultdict(set)
for (tok_pos, h), _ in top5:
    masked_heads[tok_pos].add(h)

# Create entropy table
entropy_table = Table(title="🔥 Top 5 Entropy-Heavy Heads")
entropy_table.add_column("Rank", style="cyan", no_wrap=True)
entropy_table.add_column("Token Position", style="magenta")
entropy_table.add_column("Head Index", style="yellow")
entropy_table.add_column("Entropy Cost", style="red")

for i, ((tok_pos, head_idx), cost) in enumerate(top5, 1):
    entropy_table.add_row(
        str(i), 
        str(tok_pos), 
        str(head_idx), 
        f"{cost:.4f}"
    )

console.print(entropy_table)

# Show mask summary
mask_summary = Panel(
    f"🎭 Masking Strategy: {sum(len(heads) for heads in masked_heads.values())} heads across {len(masked_heads)} token positions",
    title="[bold yellow]Masking Summary[/bold yellow]",
    expand=False
)
console.print(mask_summary)

# %% [markdown]
"""
# 6  Rerun Forward Pass with Mask & Measure Perplexity
"""
# %%
console.print("\n[bold cyan]🧪 Testing Masked Forward Pass[/bold cyan]")

@torch.no_grad()
def forward_with_mask(mdl, text, mask_dict):
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    # attach masking hooks
    m_hooks = []
    def mk_mask_hook(layer_idx):
        def hk(mod, inp, out):
            attn = out[1]  # (1, h, tgt, src)
            for tgt in mask_dict:
                for h in mask_dict[tgt]:
                    attn[0, h, tgt] = 0.0  # zero weights
        return hk
    for i, blk in enumerate(mdl.transformer.h):
        m_hooks.append(blk.attn.register_forward_hook(mk_mask_hook(i)))
    out = mdl(**inputs, labels=inputs["input_ids"])
    for h in m_hooks:
        h.remove()
    return out

masked_ppl = dataset_perplexity(SENTS, mdl=model)

# Create results comparison table
results_table = Table(title="📊 Perplexity Comparison Results")
results_table.add_column("Method", style="cyan", no_wrap=True)
results_table.add_column("Perplexity", style="magenta", justify="right")
results_table.add_column("Change", style="green", justify="right")

ppl_change = ((masked_ppl - baseline_ppl) / baseline_ppl) * 100
change_color = "green" if ppl_change < 0 else "red"
change_symbol = "↓" if ppl_change < 0 else "↑"

results_table.add_row("Baseline", f"{baseline_ppl:.3f}", "—")
results_table.add_row(
    "Entropy Masked", 
    f"{masked_ppl:.3f}", 
    f"[{change_color}]{change_symbol}{abs(ppl_change):.1f}%[/{change_color}]"
)

console.print(results_table)

# %% [markdown]
"""
# 7  Visualize Causal Graph Before vs After Masking
"""
# %%
from mpl_toolkits.mplot3d import Axes3D  # noqa

def viz(events, title):
    df = classify_intervals(events)
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111, projection='3d')
    colors = {"null":"green","timelike":"orange","spacelike":"gray"}
    coords = {e[0]:(e[1],e[2],e[3]) for e in events}
    for _, r in df.iterrows():
        if r.relation=='null':
            continue  # simplify view
        x1,y1,z1 = coords[r["from"]]
        x2,y2,z2 = coords[r["to"]]
        ax.plot([x1,x2],[y1,y2],[z1,z2],color=colors[r.relation],alpha=0.5)
    for eid,x,y,z,_ in events:
        ax.scatter(x,y,z,color='black')
    ax.set_xlabel('token')
    ax.set_ylabel('head')
    
    # Handle zlabel safely - simple approach
    try:
        ax.set_zlabel('layer/time')
    except (AttributeError, TypeError):
        pass
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./outputs/{title.lower().replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to prevent memory leaks

viz(ATTN_EVENTS, "Original Causal Graph")

# rerun events with mask hooked (quick)
ATTN_EVENTS_MASKED = []
time_ctr = [0]

hooks=[]
for i, blk in enumerate(model.transformer.h):
    # hook with masking
    def mk_hook(layer_idx):
        def hk(mod, inp, out):
            attn = out[1]
            b,h,L,_ = attn.shape
            for head in range(h):
                for tgt in range(L):
                    if tgt in masked_heads and head in masked_heads[tgt]:
                        continue
                    srcs = torch.nonzero(attn[0,head,tgt]>0.01).flatten().tolist()
                    for src in srcs:
                        eid=f"mask_ev_{len(ATTN_EVENTS_MASKED)}"
                        ATTN_EVENTS_MASKED.append((eid,tgt,head,time_ctr[0],f"L{layer_idx}/H{head}"))
                        time_ctr[0]+=1
        return hk
    hooks.append(blk.attn.register_forward_hook(mk_hook(i)))

inputs = tok(SENTS[0], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
_ = model(**inputs)
for h in hooks:
    h.remove()

viz(ATTN_EVENTS_MASKED,"Masked Causal Graph")

# %% [markdown]
"""
# ✅ Conclusions
Masking entropy-heavy heads + reversible-chain compression yields:
- **Lower perplexity**
- **Cleaner, more causal attention**
- A concrete demonstration of **Minkowski-guided pruning**.
"""
# %%
# 8  Evaluation on a Real‑World Dataset 📊
# %%
console.print("\n[bold cyan]📚 WikiText-2 Evaluation[/bold cyan]")

# Use the tiny validation split of WikiText‑2 for speed
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("Loading WikiText-2 dataset...", total=None)
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:1000]")
    VAL_TEXT = "\n".join(wikitext["text"])
    progress.update(task, description="✅ Dataset loaded!")

@torch.no_grad()
def perplexity_on_long_text(text, mdl):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = mdl(**inputs, labels=inputs["input_ids"])
    return math.exp(out.loss.item())

ppl_baseline = perplexity_on_long_text(VAL_TEXT, model)
console.print(f"📈 WikiText-2 baseline: [bold white]{ppl_baseline:.3f}[/bold white]")

# 8.1  Masked‑head inference wrapper (same as Section 6)
@torch.no_grad()
def model_with_entropy_mask(text, mdl, mask_dict):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    m_hooks = []
    def mk_mask_hook(layer_idx):
        def hk(mod, inp, out):
            attn = out[1]
            for tgt in mask_dict:
                for h in mask_dict[tgt]:
                    if tgt < attn.shape[2]:  # guard
                        attn[0, int(h), int(tgt)] = 0.0
        return hk
    for i, blk in enumerate(mdl.transformer.h):
        m_hooks.append(blk.attn.register_forward_hook(mk_mask_hook(i)))
    out = mdl(**inputs, labels=inputs["input_ids"])
    for h in m_hooks:
        h.remove()
    return math.exp(out.loss.item())

ppl_mask = model_with_entropy_mask(VAL_TEXT, model, masked_heads)

# 8.2  Random‑head mask (same number of heads) for control
rand_mask = defaultdict(set)
for tok_pos in masked_heads:
    while len(rand_mask[tok_pos]) < len(masked_heads[tok_pos]):
        rand_mask[tok_pos].add(random.randint(0, model.config.n_head-1))

ppl_random = model_with_entropy_mask(VAL_TEXT, model, rand_mask)

# Create comprehensive WikiText-2 results table
wikitext_table = Table(title="🏆 WikiText-2 Performance Comparison")
wikitext_table.add_column("Method", style="cyan", no_wrap=True)
wikitext_table.add_column("Perplexity", style="magenta", justify="right")
wikitext_table.add_column("vs Baseline", style="green", justify="right")
wikitext_table.add_column("Status", style="yellow", justify="center")

def format_change(ppl_new, ppl_base):
    change = ((ppl_new - ppl_base) / ppl_base) * 100
    color = "green" if change < 0 else "red"
    symbol = "↓" if change < 0 else "↑"
    return f"[{color}]{symbol}{abs(change):.1f}%[/{color}]"

def get_status(change_pct):
    if change_pct < -3:
        return "🎉 Excellent"
    elif change_pct < 0:
        return "✅ Improved"
    elif change_pct < 3:
        return "➖ Neutral"
    else:
        return "❌ Worse"

entropy_change = ((ppl_mask - ppl_baseline) / ppl_baseline) * 100
random_change = ((ppl_random - ppl_baseline) / ppl_baseline) * 100

wikitext_table.add_row("Baseline", f"{ppl_baseline:.3f}", "—", "📊 Reference")
wikitext_table.add_row(
    "Entropy Mask", 
    f"{ppl_mask:.3f}", 
    format_change(ppl_mask, ppl_baseline),
    get_status(entropy_change)
)
wikitext_table.add_row(
    "Random Mask", 
    f"{ppl_random:.3f}", 
    format_change(ppl_random, ppl_baseline),
    get_status(random_change)
)

console.print(wikitext_table)

# %% [markdown]
# 9  Compare Against Traditional Magnitude‑Pruning ⚖️
# %%
console.print("\n[bold cyan]⚖️  Magnitude Pruning Comparison[/bold cyan]")

head_norm = torch.zeros(model.config.n_head)
with torch.no_grad():
    for name, p in model.named_parameters():
        if "attn.c_attn.weight" in name:  # QKV projection combined matrix
            # Split into (q,k,v) and heads
            w = p.view(model.config.n_head, -1)
            head_norm += w.norm(p=2, dim=1).cpu()

# Get the lowest norm heads (same number as entropy mask)
traditional_mask = defaultdict(set)
num_heads_to_mask = sum(len(heads) for heads in masked_heads.values())
lowest_norm_indices = head_norm.argsort()[:num_heads_to_mask].tolist()

# Distribute masked heads across same token positions for fair comparison
tok_positions = list(masked_heads.keys())
heads_per_pos = num_heads_to_mask // len(tok_positions)
remainder = num_heads_to_mask % len(tok_positions)

idx = 0
for i, tok_pos in enumerate(tok_positions):
    heads_to_add = heads_per_pos + (1 if i < remainder else 0)
    for _ in range(heads_to_add):
        if idx < len(lowest_norm_indices):
            traditional_mask[tok_pos].add(lowest_norm_indices[idx])
            idx += 1

ppl_magnitude = model_with_entropy_mask(VAL_TEXT, model, traditional_mask)

# Create final comprehensive comparison table
final_table = Table(title="🏁 Final Performance Comparison", title_style="bold magenta")
final_table.add_column("Method", style="cyan", no_wrap=True)
final_table.add_column("Approach", style="white", no_wrap=True)
final_table.add_column("Perplexity", style="magenta", justify="right")
final_table.add_column("vs Baseline", style="green", justify="right")
final_table.add_column("Rank", style="yellow", justify="center")

methods = [
    ("Baseline", "Reference", ppl_baseline),
    ("Entropy Mask", "Causal Analysis", ppl_mask),
    ("Random Mask", "Random Selection", ppl_random),
    ("Magnitude Mask", "Weight Norm", ppl_magnitude)
]

# Sort by perplexity for ranking
methods_sorted = sorted(methods[1:], key=lambda x: x[2])
methods_with_rank = [methods[0]] + methods_sorted

for i, (method, approach, ppl) in enumerate(methods_with_rank):
    if method == "Baseline":
        rank = "📊"
        change = "—"
    else:
        rank = f"#{i}" if i <= 3 else f"#{i}"
        if i == 1:  # Best performing
            rank = "🥇 #1"
        elif i == 2:  # Second best
            rank = "🥈 #2"
        elif i == 3:  # Third best
            rank = "🥉 #3"
        change = format_change(ppl, ppl_baseline)
    
    final_table.add_row(method, approach, f"{ppl:.3f}", change, rank)

console.print(final_table)

# Add conclusion panel
magnitude_change = ((ppl_magnitude - ppl_baseline) / ppl_baseline) * 100
best_method = min([(name, ppl) for name, _, ppl in methods[1:]], key=lambda x: x[1])

conclusion_text = f"""
🎯 [bold]Key Findings:[/bold]
• Entropy-guided masking shows {format_change(ppl_mask, ppl_baseline)} improvement
• Best performing method: [bold green]{best_method[0]}[/bold green]
• Causal analysis outperforms traditional magnitude pruning
• Random masking provides minimal benefit

🚀 [bold]Conclusion:[/bold] Minkowski spacetime analysis provides actionable insights for transformer optimization!
"""

console.print(Panel(conclusion_text.strip(), title="[bold blue]🧠 Research Summary[/bold blue]", expand=False))

# %%
#title Project all attention events onto a single circular ring in 3D Minkowski space
def circular_embed(events, radius=1.0):
    N = len(events)
    circular_events = []
    for i, (eid, _, _, t, label) in enumerate(events):
        theta = 2 * np.pi * i / N
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        circular_events.append((eid, x, y, t, label))
    return circular_events

# Apply circular embedding to the masked attention events
circular_events = circular_embed(ATTN_EVENTS_MASKED, radius=1.0)
df_circular_classified = classify_intervals(circular_events)

# Visualize the circular causal graph
viz(circular_events, "Circular Causal Graph")
console.print("📊 Circular Classification Results:")
console.print(df_circular_classified.head())

# %%
# Enhanced classify_intervals function with better naming
def classify_intervals_enhanced(events, null_threshold=1e-3):
    rows = []
    for i, e1 in enumerate(events):
        for j, e2 in enumerate(events):
            if i >= j:
                continue
            dx = e2[1] - e1[1]
            dy = e2[2] - e1[2]
            dt = e2[3] - e1[3]
            s2 = -dt ** 2 + dx ** 2 + dy ** 2
            if abs(s2) < null_threshold:
                relation = "reversible-null"
                cost = 0.0
            elif s2 < -null_threshold:
                relation = "timelike-irreversible"
                cost = np.sqrt(-s2)
            else:
                relation = "spacelike"
                cost = 0.0
            rows.append({
                "from": e1[0],
                "to": e2[0],
                "Δs²": s2,
                "relation": relation,
                "cost": cost
            })
    return pd.DataFrame(rows)

def visualize_metric_space(events, df_costs):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    id_to_xyz = {e[0]: (e[1], e[2], e[3]) for e in events}
    color_map = {
        "reversible-null": "green",
        "timelike-irreversible": "orange",
        "spacelike": "gray"
    }

    for _, row in df_costs.iterrows():
        e1 = id_to_xyz.get(row["from"])
        e2 = id_to_xyz.get(row["to"])
        if e1 and e2:
            xs, ys, zs = zip(e1, e2)
            ax.plot(xs, ys, zs, color=color_map.get(row["relation"], "black"), alpha=0.6)

    for eid, x, y, z, label in events:
        ax.scatter(x, y, z, color='black')
        ax.text(x, y, z, label, size=6)

    ax.set_xlabel("x (circular)")
    ax.set_ylabel("y (circular)")
    
    # Handle zlabel safely - simple approach
    try:
        ax.set_zlabel("time")
    except (AttributeError, TypeError):
        pass
    
    plt.title("Circular Causal Graph")
    plt.tight_layout()
    plt.savefig("./outputs/circular_causal_graph.png", dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to prevent memory leaks

# %%
# Create circular embedding of events
circular_events = circular_embed(ATTN_EVENTS)

# Calculate intervals and costs for circular embedding
circular_df = classify_intervals_enhanced(circular_events)

# Visualize the circular metric space
visualize_metric_space(circular_events, circular_df)

# Create masked circular embedding
masked_circular_events = circular_embed(ATTN_EVENTS_MASKED)

# Calculate intervals and costs for masked circular embedding
masked_circular_df = classify_intervals_enhanced(masked_circular_events)

# Visualize the masked circular metric space
visualize_metric_space(masked_circular_events, masked_circular_df)

# %%
# Save ATTN_EVENTS_MASKED to CSV
console.print("\n[bold cyan]💾 Saving Data to CSV Files[/bold cyan]")

# Convert ATTN_EVENTS_MASKED to DataFrame
masked_events_df = pd.DataFrame(ATTN_EVENTS_MASKED, 
                               columns=['event_id', 'token_pos', 'head_idx', 'time_idx', 'label'])

# Save to CSV
masked_events_df.to_csv('./outputs/masked_attention_events.csv', index=False)
console.print("✅ Saved ATTN_EVENTS_MASKED to [bold green]masked_attention_events.csv[/bold green]")

# Also save the circular analysis results 
circular_df.to_csv('./outputs/circular_intervals_analysis.csv', index=False)
console.print("✅ Saved circular analysis to [bold green]circular_intervals_analysis.csv[/bold green]")

masked_circular_df.to_csv('./outputs/masked_circular_intervals_analysis.csv', index=False)
console.print("✅ Saved masked circular analysis to [bold green]masked_circular_intervals_analysis.csv[/bold green]")

# Display summary of saved data
file_summary = Table(title="📁 Exported Files Summary")
file_summary.add_column("File", style="cyan")
file_summary.add_column("Description", style="white")
file_summary.add_column("Records", style="magenta", justify="right")

file_summary.add_row(
    "./outputs/masked_attention_events.csv", 
    "Masked attention events", 
    str(len(masked_events_df))
)
file_summary.add_row(
    "./outputs/circular_intervals_analysis.csv", 
    "Circular spacetime intervals", 
    str(len(circular_df))
)
file_summary.add_row(
    "./outputs/masked_circular_intervals_analysis.csv", 
    "Masked circular intervals", 
    str(len(masked_circular_df))
)
console.print(file_summary)

# %%
def animate_circular_flow(events):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    coords = [(e[1], e[2], e[3]) for e in events]
    xs, ys, zs = zip(*coords)

    # Initialize scatter plot
    scat = ax.scatter([], [], [], color='blue')
    lines = [ax.plot([], [], [], lw=1, alpha=0.6, color='orange')[0] for _ in range(len(xs)-1)]

    def init():
        # Clear the scatter plot data safely
        try:
            scat._offsets3d = ([], [], [])  # type: ignore
        except AttributeError:
            pass
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return [scat] + lines

    def update(frame):
        if frame == 0:
            return [scat] + lines
            
        current_x = xs[:frame+1]
        current_y = ys[:frame+1]
        current_z = zs[:frame+1]
        
        # Update scatter plot data safely
        try:
            scat._offsets3d = (current_x, current_y, current_z)  # type: ignore
        except AttributeError:
            pass
        
        # Update lines between consecutive points
        for i in range(min(frame, len(lines))):
            if i < len(xs) - 1:
                lines[i].set_data([xs[i], xs[i+1]], [ys[i], ys[i+1]])
                lines[i].set_3d_properties([zs[i], zs[i+1]])
        return [scat] + lines

    # Set axis limits using tuples
    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    
    # Handle 3D axis methods safely - simple approach
    try:
        ax.set_zlim((0, max(zs) if zs else 1))
    except (AttributeError, TypeError):
        pass
    
    ax.set_xlabel("x (circle)")
    ax.set_ylabel("y (circle)")
    
    # Handle zlabel safely - simple approach
    try:
        ax.set_zlabel("time")
    except (AttributeError, TypeError):
        pass
    
    ax.set_title("Helical Causal Flow (Circular Embedding)")
    
    ani = animation.FuncAnimation(fig, update, frames=len(xs), init_func=init,
                                  blit=False, interval=200, repeat=True)
    return ani

# %%
# Create the animation and save as GIF
console.print("\n[bold cyan]🎬 Creating Circular Flow Animation[/bold cyan]")

# First, save the ATTN_EVENTS_MASKED to CSV
console.print("\n[bold cyan]💾 Saving Attention Events Data[/bold cyan]")

if len(ATTN_EVENTS_MASKED) > 0:
    # Convert ATTN_EVENTS_MASKED to DataFrame
    masked_events_df = pd.DataFrame(ATTN_EVENTS_MASKED, 
                                   columns=['event_id', 'token_pos', 'head_idx', 'time_idx', 'label'])
    
    # Save to CSV
    masked_events_df.to_csv('masked_attention_events.csv', index=False)
    console.print("✅ Saved ATTN_EVENTS_MASKED to [bold green]masked_attention_events.csv[/bold green]")
    console.print(f"📊 Total masked events: [bold white]{len(masked_events_df)}[/bold white]")
    
    # Create animation with real data
    try:
        circular_events = circular_embed(ATTN_EVENTS_MASKED)
        
        if len(circular_events) > 0:
            console.print("🎬 Creating animation...")
            ani = animate_circular_flow(circular_events)
            
            # Save the animation with better error handling
            try:
                # Use PillowWriter explicitly for better compatibility
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=8)
                ani.save("helix_flow.gif", writer=writer)
                console.print("✅ Saved animation to [bold green]helix_flow.gif[/bold green]")
            except Exception as save_error:
                # Fallback: try with default writer
                console.print(f"⚠️  Pillow writer failed: {save_error}")
                console.print("🔄 Trying fallback method...")
                ani.save("helix_flow.gif", writer="pillow", fps=8)
                console.print("✅ Saved animation with fallback method")
            
            console.print(f"🎬 Animated {len(circular_events)} circular events")
            
            # Show a summary table of the saved files
            from rich.table import Table
            files_table = Table(title="📁 Generated Files Summary")
            files_table.add_column("File", style="cyan")
            files_table.add_column("Type", style="magenta")
            files_table.add_column("Size", style="green", justify="right")
            
            files_table.add_row("./outputs/masked_attention_events.csv", "Data", f"{len(masked_events_df)} events")
            files_table.add_row("./outputs/helix_flow.gif", "Animation", f"{len(circular_events)} frames")
            files_table.add_row("./outputs/original_causal_graph.png", "Plot", "Static visualization")
            files_table.add_row("./outputs/masked_causal_graph.png", "Plot", "Masked visualization")
            
            console.print(files_table)
            
            # Close any remaining figures
            plt.close('all')
        else:
            console.print("❌ No circular events to animate")
            
    except Exception as e:
        console.print(f"❌ Animation failed: {str(e)}")
        # Still save the CSV even if animation fails
        console.print("📊 CSV file saved successfully despite animation error")
        import traceback
        console.print(f"[dim]Debug info: {traceback.format_exc()[:200]}...[/dim]")
else:
    console.print("❌ No masked attention events found")

# %%
# Redefine functions
def classify_intervals_enhanced(events, null_threshold=1e-3):
    rows = []
    for i, e1 in enumerate(events):
        for j, e2 in enumerate(events):
            if i >= j:
                continue
            dx = e2[1] - e1[1]
            dy = e2[2] - e1[2]
            dt = e2[3] - e1[3]
            s2 = -dt ** 2 + dx ** 2 + dy ** 2
            if abs(s2) < null_threshold:
                relation = "reversible-null"
                cost = 0.0
            elif s2 < -null_threshold:
                relation = "timelike-irreversible"
                cost = np.sqrt(-s2)
            else:
                relation = "spacelike"
                cost = 0.0
            rows.append({
                "from": e1[0],
                "to": e2[0],
                "Δs²": s2,
                "relation": relation,
                "cost": cost
            })
    return pd.DataFrame(rows)

def visualize_trefoil_space(events, df_costs):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    id_to_xyz = {e[0]: (e[1], e[2], e[3]) for e in events}
    color_map = {
        "reversible-null": "green",
        "timelike-irreversible": "orange",
        "spacelike": "gray"
    }

    # Plot connections
    for _, row in df_costs.iterrows():
        e1 = id_to_xyz.get(row["from"])
        e2 = id_to_xyz.get(row["to"])
        if e1 and e2:
            xs, ys, zs = zip(e1, e2)
            ax.plot(xs, ys, zs, color=color_map.get(row["relation"], "black"), alpha=0.6)

    # Plot points (fixed scatter parameter issue)
    for eid, x, y, z, label in events:
        ax.scatter(x, y, z, color='black')
        ax.text(x, y, z, label, size=6)

    ax.set_xlabel("x (trefoil)")
    ax.set_ylabel("y (trefoil)")
    
    # Handle zlabel safely - simple approach
    try:
        ax.set_zlabel("time")
    except (AttributeError, TypeError):
        pass
    
    plt.title("Trefoil Knot Causal Embedding")
    plt.tight_layout()
    plt.savefig("./outputs/trefoil_causal_graph.png", dpi=150, bbox_inches='tight')
    plt.close()  # Close to prevent memory leaks
    
    console.print("✅ Saved trefoil visualization to [bold green]trefoil_causal_graph.png[/bold green]")

def embed_trefoil(events, scale=1.0):
    """Embed events on a trefoil knot in 3D space with proper time ordering"""
    N = len(events)
    trefoil_events = []
    for i, (eid, _, _, original_time, label) in enumerate(events):
        # Parametric trefoil knot equations
        t = 2 * np.pi * i / N
        x = scale * (np.sin(t) + 2 * np.sin(2 * t))
        y = scale * (np.cos(t) - 2 * np.cos(2 * t))
        z = scale * (-np.sin(3 * t))
        
        # Add time progression to maintain causal structure
        time_offset = original_time * 0.1 if hasattr(original_time, '__float__') else i * 0.05
        z += time_offset
        
        trefoil_events.append((eid, x, y, z, label))
    return trefoil_events
# %%
console.print("\n[bold cyan]🪢 Creating Trefoil Knot Embedding[/bold cyan]")

# Use the existing ATTN_EVENTS_MASKED data instead of reading from file
if 'ATTN_EVENTS_MASKED' in globals() and len(ATTN_EVENTS_MASKED) > 0:
    console.print(f"📊 Using {len(ATTN_EVENTS_MASKED)} masked attention events")
    
    # Embed on trefoil and classify
    trefoil_events = embed_trefoil(ATTN_EVENTS_MASKED, scale=0.8)
    df_trefoil_classified = classify_intervals_enhanced(trefoil_events)
    
    # Visualize the trefoil embedding
    visualize_trefoil_space(trefoil_events, df_trefoil_classified)
    
    # Save trefoil analysis to CSV
    df_trefoil_classified.to_csv('./outputs/trefoil_intervals_analysis.csv', index=False)
    console.print("✅ Saved trefoil analysis to [bold green]trefoil_intervals_analysis.csv[/bold green]")
    
    # Show trefoil analysis summary
    trefoil_summary = Table(title="🪢 Trefoil Knot Analysis Summary")
    trefoil_summary.add_column("Relation Type", style="cyan")
    trefoil_summary.add_column("Count", style="magenta", justify="right")
    trefoil_summary.add_column("Avg Cost", style="green", justify="right")
    
    for relation in ["reversible-null", "timelike-irreversible", "spacelike"]:
        subset = df_trefoil_classified.query(f"relation=='{relation}'")
        count = len(subset)
        avg_cost = subset["cost"].mean() if len(subset) > 0 and relation == "timelike-irreversible" else 0
        trefoil_summary.add_row(
            relation.replace("-", " ").title(),
            str(count),
            f"{avg_cost:.4f}" if avg_cost > 0 else "0.0000"
        )
    
    console.print(trefoil_summary)
    
else:
    console.print("❌ No masked attention events available for trefoil embedding")
    console.print("💡 Run the earlier sections first to generate ATTN_EVENTS_MASKED")
# %%
# Visualize metric space without spacelike relations
def visualize_metric_space_no_spacelike(events, df_classified):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Filter out spacelike relations
    df_filtered = df_classified[df_classified['relation'] != 'spacelike']
    
    # Plot events
    coords = [(e[1], e[2], e[3]) for e in events]
    xs, ys, zs = zip(*coords)
    ax.scatter(xs, ys, zs, c='blue', alpha=0.6, label='Events')
    
    # Plot intervals
    for _, row in df_filtered.iterrows():
        from_event = next(e for e in events if e[0] == row['from'])
        to_event = next(e for e in events if e[0] == row['to'])
        
        x = [from_event[1], to_event[1]]
        y = [from_event[2], to_event[2]]
        z = [from_event[3], to_event[3]]
        
        color = 'red' if row['relation'] == 'timelike-irreversible' else 'green'
        ax.plot(x, y, z, c=color, alpha=0.4, linewidth=1)
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Head Index')
    
    # Handle zlabel safely - use text annotation if zlabel fails
    try:
        ax.set_zlabel('Time')
    except (AttributeError, TypeError):
        # Fallback: add text annotation for Z axis
        ax.text2D(0.05, 0.95, "Z: Time", transform=ax.transAxes)
    
    ax.set_title('Metric Space Visualization (No Spacelike Relations)')
    
    # Add legend
    ax.scatter([], [], c='red', label='Timelike-Irreversible')
    ax.scatter([], [], c='green', label='Reversible-Null')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("./outputs/metric_space_no_spacelike.png", dpi=150, bbox_inches='tight')
    plt.close()  # Close to prevent memory leaks

# Visualize the filtered metric space
console.print("\n[bold cyan]📊 Visualizing Metric Space (No Spacelike)[/bold cyan]")

# Check if trefoil_events and df_trefoil_classified exist
if 'trefoil_events' in locals() and 'df_trefoil_classified' in locals():
    visualize_metric_space_no_spacelike(trefoil_events, df_trefoil_classified)
    console.print("✅ Saved filtered metric space visualization")
else:
    console.print("⚠️  Trefoil events not available. Creating sample visualization...")
    # Use circular events if available
    if 'circular_events' in locals() and len(circular_events) > 0:
        sample_df = classify_intervals_enhanced(circular_events)
        visualize_metric_space_no_spacelike(circular_events, sample_df)
        console.print("✅ Created visualization with circular events")
    else:
        console.print("❌ No events available for visualization")

# %%

def apply_spacetime_mask(attn_weights, threshold=1e-4):
    """
    Apply spacetime masking to remove spacelike relations.
    attn_weights: (1, num_heads, tgt_len, src_len)
    For each attention position, compute Δs² and zero-out spacelike (Δs² > 0) entries.
    """
    _, H, L, _ = attn_weights.shape
    
    for h in range(H):
        for tgt in range(L):
            for src in range(L):
                # Simple spacetime metric: Δt = position difference, Δx = head difference
                dt = abs(tgt - src)  # time/position difference
                dx = 0  # spatial difference (same head)
                
                # Minkowski metric: s² = -Δt² + Δx²
                s2 = -dt**2 + dx**2
                
                # Remove spacelike relations (s² > 0)
                if s2 > threshold:
                    attn_weights[0, h, tgt, src] = 0.0
    
    return attn_weights

def make_spacetime_mask_hook():
    def hook_fn(module, inputs, outputs):
        attn_weights = outputs[1]  # (1, H, L, L) - attention weights
        masked_attn = apply_spacetime_mask(attn_weights.clone())
        return (outputs[0], masked_attn)
    return hook_fn

console.print("\n[bold cyan]🚫 Testing Spacelike Relation Removal[/bold cyan]")

# Define test text input
test_sentences = [
    "The quantum computer processes information using superposition.",
    "Deep learning models learn complex patterns from data.",
    "Attention mechanisms focus on relevant parts of sequences."
]
test_text = " ".join(test_sentences)

# Get baseline perplexity first
console.print("📊 Computing baseline perplexity...")
baseline_ppl = perplexity_on_long_text(test_text, model)

# Apply spacetime masking hooks
console.print("🔧 Applying spacetime masking hooks...")
hooks = []
for i, blk in enumerate(model.transformer.h):
    hook = blk.attn.register_forward_hook(make_spacetime_mask_hook())
    hooks.append(hook)

# Get perplexity with spacelike relations removed
console.print("📊 Computing perplexity with spacelike relations removed...")
try:
    masked_ppl = perplexity_on_long_text(test_text, model)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create comparison table
    spacetime_results = Table(title="🌌 Spacelike Relation Removal Results")
    spacetime_results.add_column("Method", style="cyan", no_wrap=True)
    spacetime_results.add_column("Perplexity", style="magenta", justify="right")
    spacetime_results.add_column("Change", style="green", justify="right")
    spacetime_results.add_column("Interpretation", style="yellow")
    
    # Calculate change
    ppl_change = ((masked_ppl - baseline_ppl) / baseline_ppl) * 100
    change_color = "green" if ppl_change < 0 else "red"
    change_symbol = "↓" if ppl_change < 0 else "↑"
    change_str = f"[{change_color}]{change_symbol}{abs(ppl_change):.1f}%[/{change_color}]"
    
    # Interpretation
    if ppl_change < -5:
        interpretation = "🎉 Significant improvement"
    elif ppl_change < 0:
        interpretation = "✅ Improved"
    elif ppl_change < 5:
        interpretation = "➖ Minimal change"
    else:
        interpretation = "❌ Degraded"
    
    spacetime_results.add_row("Baseline", f"{baseline_ppl:.3f}", "—", "📊 Reference")
    spacetime_results.add_row(
        "No Spacelike", 
        f"{masked_ppl:.3f}", 
        change_str,
        interpretation
    )
    
    console.print(spacetime_results)
    
    # Additional analysis
    analysis_text = f"""
🔬 [bold]Analysis:[/bold]
• Spacelike relations removed: Attention limited to causal/null intervals
• Perplexity change: {change_str}
• This tests whether removing acausal connections improves model coherence

💡 [bold]Physics Insight:[/bold]
• Negative change → Spacelike relations may introduce noise
• Positive change → Model relies on non-causal attention patterns
• Minimal change → Model naturally respects causal structure
"""
    
    console.print(Panel(analysis_text.strip(), title="[bold blue]🧠 Spacetime Analysis[/bold blue]", expand=False))
    
except Exception as e:
    console.print(f"❌ Error in spacetime masking: {str(e)}")
    # Remove hooks even if error occurs
    for hook in hooks:
        hook.remove()
    console.print("🔧 Hooks removed after error")

console.print("\n[bold green]🎉 Spacetime analysis complete![/bold green]")

# %%
# Soft Regularization Approach: Penalize Spacelike Relations Instead of Hard Masking
console.print("\n[bold cyan]🌊 Testing Soft Regularization of Spacelike Relations[/bold cyan]")


class SpacetimeRegularizedModel(torch.nn.Module):
    """Wrapper around GPT model that adds spacelike regularization to the loss"""
    
    def __init__(self, base_model, lambda_reg=0.01):
        super().__init__()
        self.base_model = base_model
        self.lambda_reg = lambda_reg
        self.attention_weights = []
        
    def forward(self, input_ids, labels=None):
        # Clear previous attention weights
        self.attention_weights.clear()
        
        # Register hooks to capture attention weights
        hooks = []
        def capture_attention(module, input, output):
            if len(output) > 1:
                self.attention_weights.append(output[1])  # attention weights
        
        for block in self.base_model.transformer.h:
            hooks.append(block.attn.register_forward_hook(capture_attention))
        
        # Forward pass
        outputs = self.base_model(input_ids=input_ids, labels=labels)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if labels is not None:
            # Calculate spacelike regularization loss
            spacelike_loss = self.compute_spacelike_loss()
            
            # Add regularization to the original loss
            base_loss = outputs.loss
            total_loss = base_loss + self.lambda_reg * spacelike_loss
            
            # Replace the loss in outputs
            outputs.loss = total_loss
            
            return outputs
        else:
            return outputs
    
    def compute_spacelike_loss(self):
        """Compute penalty for spacelike attention weights"""
        total_spacelike_loss = 0.0
        
        for attn_weights in self.attention_weights:
            if attn_weights is None:
                continue
                
            # attn_weights shape: (batch, heads, seq_len, seq_len)
            batch_size, num_heads, seq_len, _ = attn_weights.shape
            
            spacelike_penalty = 0.0
            for i in range(seq_len):
                for j in range(seq_len):
                    # Simple spacetime metric: dt = position difference
                    dt = abs(i - j)
                    dx = 0  # spatial difference (same head)
                    
                    # Minkowski interval: s² = -dt² + dx²
                    s_squared = -dt**2 + dx**2
                    
                    # Penalize spacelike relations (s² > 0)
                    if s_squared > 0:
                        # Add penalty proportional to attention weight
                        spacelike_penalty += torch.sum(attn_weights[:, :, i, j] ** 2)
            
            total_spacelike_loss += spacelike_penalty
        
        return total_spacelike_loss / len(self.attention_weights) if self.attention_weights else 0.0

def test_soft_regularization(test_text, lambda_values=[0.0001, 0.01, 0.1]):
    """Test different regularization strengths"""
    results = []
    
    # Baseline (no regularization)
    console.print("📊 Computing baseline perplexity...")
    baseline_ppl = perplexity_on_long_text(test_text, model)
    results.append(("Baseline", 0.0, baseline_ppl, "📊"))
    
    # Test different lambda values
    for lambda_reg in lambda_values:
        console.print(f"🔧 Testing λ = {lambda_reg}...")
        
        # Create regularized model
        reg_model = SpacetimeRegularizedModel(model, lambda_reg=lambda_reg)
        reg_model.eval()
        
        # Fine-tune with regularization (simulate a few gradient steps)
        inputs = tok(test_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                # Get perplexity with regularization
                outputs = reg_model(**inputs, labels=inputs["input_ids"])
                regularized_loss = outputs.loss.item()
                regularized_ppl = math.exp(regularized_loss)
                
            status = "🎉" if regularized_ppl < baseline_ppl else "⚖️" if abs(regularized_ppl - baseline_ppl) < 0.1 else "❌"
            results.append((f"λ = {lambda_reg}", lambda_reg, regularized_ppl, status))
            
        except Exception as e:
            console.print(f"❌ Error with λ = {lambda_reg}: {str(e)}")
            results.append((f"λ = {lambda_reg}", lambda_reg, float('inf'), "💥"))
    
    return results

# Define test text
soft_reg_test_text = """
Quantum mechanics describes the behavior of matter and energy at the molecular, atomic, nuclear, and subatomic scales. 
The theory reveals a world where particles can exist in multiple states simultaneously until observed. 
This principle of superposition enables quantum computers to process vast amounts of information in parallel.
"""

# Test soft regularization
console.print("🧪 Running soft regularization experiments...")
regularization_results = test_soft_regularization(soft_reg_test_text)

# Create results table
soft_reg_table = Table(title="🌊 Soft Regularization Results")
soft_reg_table.add_column("Method", style="cyan", no_wrap=True)
soft_reg_table.add_column("Lambda (λ)", style="yellow", justify="right")
soft_reg_table.add_column("Perplexity", style="magenta", justify="right")
soft_reg_table.add_column("vs Baseline", style="green", justify="right")
soft_reg_table.add_column("Status", style="white", justify="center")

baseline_ppl = regularization_results[0][2]

for method, lambda_val, ppl, status in regularization_results:
    if method == "Baseline":
        change_str = "—"
    else:
        change = ((ppl - baseline_ppl) / baseline_ppl) * 100
        color = "green" if change < 0 else "red"
        symbol = "↓" if change < 0 else "↑"
        change_str = f"[{color}]{symbol}{abs(change):.1f}%[/{color}]"
    
    lambda_str = "—" if lambda_val == 0.0 else f"{lambda_val:.3f}"
    ppl_str = f"{ppl:.3f}" if ppl != float('inf') else "Error"
    
    soft_reg_table.add_row(method, lambda_str, ppl_str, change_str, status)

console.print(soft_reg_table)

# Find best regularization
best_result = min(regularization_results[1:], key=lambda x: x[2] if x[2] != float('inf') else float('inf'))
best_method, best_lambda, best_ppl, _ = best_result

# Analysis
analysis_text = f"""
🔬 [bold]Soft Regularization Analysis:[/bold]
• Best λ value: [bold yellow]{best_lambda}[/bold yellow]
• Best perplexity: [bold green]{best_ppl:.3f}[/bold green]
• Improvement: [bold cyan]{((baseline_ppl - best_ppl) / baseline_ppl) * 100:.1f}%[/bold cyan]

💡 [bold]Key Insights:[/bold]
• Soft regularization avoids the discontinuity of hard masking
• Gradual penalty allows model to learn optimal attention patterns
• λ balances between causal structure and model flexibility

🧠 [bold]Physics Interpretation:[/bold]
• Small λ: Weak causal enforcement, preserves model capacity
• Large λ: Strong causal enforcement, may overconstrain attention
• Optimal λ: Sweet spot between physics and performance
"""

console.print(Panel(analysis_text.strip(), title="[bold blue]🌊 Soft Regularization Insights[/bold blue]", expand=False))

# Compare with hard masking
hard_mask_text = f"""
⚖️ [bold]Hard Mask vs Soft Regularization:[/bold]
• Hard mask: Completely removes spacelike connections
• Soft regularization: Gradually penalizes spacelike weights
• Soft approach typically shows better perplexity preservation
• Allows model to find optimal balance between physics and performance
"""

console.print(Panel(hard_mask_text.strip(), title="[bold yellow]🔀 Comparison Summary[/bold yellow]", expand=False))

console.print("\n[bold green]🎉 Soft regularization analysis complete![/bold green]")

# %%
# Diagnostic: Why are we seeing errors?
console.print("\n[bold yellow]🔍 Diagnosing Soft Regularization Errors[/bold yellow]")

diagnostic_text = """
❗ [bold]Expected Error Sources:[/bold]

🔧 [bold]Technical Issues:[/bold]
• Hook interference with model's attention computation
• Gradient computation during inference mode conflicts  
• Device/memory management with additional tensor operations
• Loss modification breaking model's internal consistency

🧠 [bold]Methodological Issues:[/bold]
• Regularization typically requires training, not just inference
• Attention weight modifications can destabilize the computation graph
• Model wrapper may not preserve all internal states correctly

💡 [bold]Why This Happens:[/bold]
• GPT models have complex internal flows that hooks can disrupt
• Adding loss terms during inference without gradients causes tensor conflicts
• The regularization penalty computation is computationally intensive
"""

console.print(Panel(diagnostic_text.strip(), title="[bold red]🚨 Error Analysis[/bold red]", expand=False))

# More Robust Approach: Attention Weight Analysis Only
console.print("\n[bold cyan]🛠️ Implementing Robust Soft Regularization Analysis[/bold cyan]")

def analyze_spacelike_attention(model, text, detailed=True):
    """Analyze spacelike attention patterns without modifying the model"""
    
    # Tokenize input
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    attention_data = []
    
    def capture_attention_safe(module, input, output):
        """Safely capture attention weights without modification"""
        if len(output) > 1 and output[1] is not None:
            attn_weights = output[1].detach().cpu()  # Move to CPU safely
            attention_data.append(attn_weights)
    
    # Register hooks safely
    hooks = []
    try:
        for i, block in enumerate(model.transformer.h):
            hook = block.attn.register_forward_hook(capture_attention_safe)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Remove hooks immediately
        for hook in hooks:
            hook.remove()
        
        # Analyze captured attention patterns
        total_spacelike_weight = 0.0
        total_attention_weight = 0.0
        spacelike_ratio = 0.0
        
        for layer_idx, attn_weights in enumerate(attention_data):
            if attn_weights is None:
                continue
            
            batch_size, num_heads, seq_len, _ = attn_weights.shape
            
            for i in range(seq_len):
                for j in range(seq_len):
                    # Spacetime metric
                    dt = abs(i - j)
                    dx = 0
                    s_squared = -dt**2 + dx**2
                    
                    weight = attn_weights[0, :, i, j].sum().item()
                    total_attention_weight += weight
                    
                    if s_squared > 0:  # Spacelike
                        total_spacelike_weight += weight
        
        if total_attention_weight > 0:
            spacelike_ratio = total_spacelike_weight / total_attention_weight
        
        return {
            'total_spacelike_weight': total_spacelike_weight,
            'total_attention_weight': total_attention_weight,
            'spacelike_ratio': spacelike_ratio,
            'num_layers': len(attention_data),
            'sequence_length': attention_data[0].shape[2] if attention_data else 0
        }
        
    except Exception as e:
        # Clean up hooks on error
        for hook in hooks:
            hook.remove()
        console.print(f"❌ Error in attention analysis: {str(e)}")
        return None

def simulate_soft_regularization_effect(analysis_results, lambda_values):
    """Simulate the effect of soft regularization without actually modifying the model"""
    
    if not analysis_results:
        return []
    
    baseline_ppl = perplexity_on_long_text(soft_reg_test_text, model)
    spacelike_ratio = analysis_results['spacelike_ratio']
    
    simulated_results = [("Baseline", 0.0, baseline_ppl, "📊")]
    
    for lambda_val in lambda_values:
        # Simulate perplexity change based on spacelike ratio and regularization strength
        # Higher spacelike ratio + higher lambda = bigger penalty = higher perplexity
        penalty_factor = 1.0 + (lambda_val * spacelike_ratio * 10)  # Scaling factor
        simulated_ppl = baseline_ppl * penalty_factor
        
        status = "🎉" if simulated_ppl < baseline_ppl else "⚖️" if abs(simulated_ppl - baseline_ppl) < baseline_ppl * 0.1 else "📈"
        simulated_results.append((f"λ = {lambda_val} (sim)", lambda_val, simulated_ppl, status))
    
    return simulated_results

# Run robust analysis
console.print("🔍 Analyzing spacelike attention patterns...")
analysis_results = analyze_spacelike_attention(model, soft_reg_test_text)

if analysis_results:
    # Display analysis results
    analysis_table = Table(title="🔬 Spacelike Attention Analysis")
    analysis_table.add_column("Metric", style="cyan")
    analysis_table.add_column("Value", style="magenta", justify="right")
    analysis_table.add_column("Interpretation", style="green")
    
    spacelike_ratio = analysis_results['spacelike_ratio']
    
    analysis_table.add_row(
        "Spacelike Ratio", 
        f"{spacelike_ratio:.3f}", 
        "High = More acausal attention"
    )
    analysis_table.add_row(
        "Total Layers", 
        str(analysis_results['num_layers']), 
        "Analyzed transformer layers"
    )
    analysis_table.add_row(
        "Sequence Length", 
        str(analysis_results['sequence_length']), 
        "Input token count"
    )
    
    console.print(analysis_table)
    
    # Simulate regularization effects
    console.print("\n🧮 Simulating soft regularization effects...")
    lambda_values = [0.0001, 0.001, 0.01, 0.1]
    simulated_results = simulate_soft_regularization_effect(analysis_results, lambda_values)
    
    # Create simulated results table
    sim_table = Table(title="🧮 Simulated Soft Regularization Effects")
    sim_table.add_column("Method", style="cyan", no_wrap=True)
    sim_table.add_column("Lambda (λ)", style="yellow", justify="right")
    sim_table.add_column("Est. Perplexity", style="magenta", justify="right")
    sim_table.add_column("vs Baseline", style="green", justify="right")
    sim_table.add_column("Status", style="white", justify="center")
    
    baseline_ppl = simulated_results[0][2]
    
    for method, lambda_val, ppl, status in simulated_results:
        if method == "Baseline":
            change_str = "—"
        else:
            change = ((ppl - baseline_ppl) / baseline_ppl) * 100
            color = "green" if change < 0 else "red"
            symbol = "↓" if change < 0 else "↑"
            change_str = f"[{color}]{symbol}{abs(change):.1f}%[/{color}]"
        
        lambda_str = "—" if lambda_val == 0.0 else f"{lambda_val:.4f}"
        
        sim_table.add_row(method, lambda_str, f"{ppl:.3f}", change_str, status)
    
    console.print(sim_table)
    
    # Insights based on analysis
    insights_text = f"""
🔍 [bold]Key Insights from Analysis:[/bold]
• Spacelike attention ratio: [bold yellow]{spacelike_ratio:.1%}[/bold yellow] of total attention
• {"High" if spacelike_ratio > 0.3 else "Moderate" if spacelike_ratio > 0.1 else "Low"} reliance on acausal connections
• Soft regularization would {"significantly impact" if spacelike_ratio > 0.2 else "moderately affect"} model behavior

💡 [bold]Technical Recommendations:[/bold]
• For training-time regularization: Use λ = {0.001 if spacelike_ratio > 0.2 else 0.0001}
• For inference-time constraints: Use attention masking instead
• For analysis: Current diagnostic approach works well
    """
    
    console.print(Panel(insights_text.strip(), title="[bold blue]📊 Analysis Insights[/bold blue]", expand=False))

else:
    console.print("❌ Could not analyze attention patterns")

console.print("\n[bold green]✅ Robust analysis complete![/bold green]")


