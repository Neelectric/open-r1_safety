# %%
# !pip install -q transformers accelerate
# !pip install -U -q jupyter ipywidgets

# %%
import math
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model_name = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # for batching

# # Tiny toy corpora
# protected_french_texts = [
#     "Ceci est une phrase française simple.",
#     "Le chat noir dort sur le canapé.",
#     "La météo est agréable aujourd'hui à Paris.",
#     "Les réseaux de neurones profonds apprennent des représentations complexes.",
# ] * 200  # repeat to get more samples

# new_english_texts = [
#     "This is a simple English sentence.",
#     "The neural network is fine-tuned on a new dataset.",
#     "We evaluate catastrophic forgetting in this experiment.",
#     "Language models are tested on multiple capabilities.",
# ] * 200

# !wget -q -O petit_prince.txt https://www.gutenberg.org/cache/epub/70167/pg70167.txt
# !wget -q -O alice.txt https://www.gutenberg.org/cache/epub/11/pg11.txt

with open("petit_prince.txt") as f:
    protected_french_texts = [line.strip() for line in f if len(line.strip()) > 20]

with open("alice.txt") as f:
    new_english_texts = [line.strip() for line in f if len(line.strip()) > 20]

# Train / test splits
split_f = int(0.8 * len(protected_french_texts))
split_e = int(0.8 * len(new_english_texts))

french_train = protected_french_texts[:split_f]
french_test  = protected_french_texts[split_f:]

english_train = new_english_texts[:split_e]
english_test  = new_english_texts[split_e:]


# %%
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class LineByLineLMDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=64):
        self.examples = []
        for t in texts:
            ids = tokenizer(
                t,
                truncation=True,
                max_length=block_size,
                return_attention_mask=False,
                return_tensors="pt",
            )["input_ids"][0]
            if ids.numel() > 1:
                self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    batch = [b for b in batch if b.numel() > 1]
    max_len = max(x.size(0) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded = torch.cat([x, x.new_full((pad_len,), tokenizer.pad_token_id)])
        mask = torch.cat([torch.ones_like(x), torch.zeros(pad_len, dtype=torch.long)])
        lab = padded.clone()
        lab[mask == 0] = -100  # ignore padding in loss
        input_ids.append(padded)
        attention_mask.append(mask)
        labels.append(lab)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }

block_size = 64
batch_size = 8

french_train_ds  = LineByLineLMDataset(french_train, tokenizer, block_size)
french_test_ds   = LineByLineLMDataset(french_test,  tokenizer, block_size)
english_train_ds = LineByLineLMDataset(english_train, tokenizer, block_size)
english_test_ds  = LineByLineLMDataset(english_test,  tokenizer, block_size)

eng_loader = DataLoader(english_train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
fr_loader  = DataLoader(french_train_ds,  batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

len(english_train_ds), len(french_train_ds)


# %%
@torch.no_grad()
def eval_ppl(model, dataset, name, batch_size_eval=8):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=collate_fn)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss * batch["attention_mask"].sum()
        total_loss += loss.item()
        total_tokens += batch["attention_mask"].sum().item()
    ppl = math.exp(total_loss / total_tokens)
    print(f"{name} perplexity: {ppl:.3f}")
    model.train()
    return ppl


# %%
from torch.optim import AdamW

def run_baseline_adam(num_epochs=2, lr=1e-5):
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    print(f"Before opt:")
    eng_ppl = eval_ppl(model, english_test_ds, "English new")
    fr_ppl  = eval_ppl(model, french_test_ds,  "French protected")


    print("=== Baseline Adam: train on English only ===")
    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch} Step {step+1}] loss_new = {loss.item():.4f}")

        print(f"Epoch {epoch} evaluation:")
        eng_ppl = eval_ppl(model, english_test_ds, "English new")
        fr_ppl  = eval_ppl(model, french_test_ds,  "French protected")
    return model

baseline_model = run_baseline_adam(num_epochs=4, lr=1e-5)


# %%
def estimate_fisher_on_french(model, num_batches=200):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    fisher = {p: torch.zeros_like(p.data) for p in params}

    loader = DataLoader(french_train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    it = iter(loader)
    for i in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        for p in params:
            if p.grad is None:
                continue
            fisher[p] += p.grad.data.pow(2)
    for p in params:
        fisher[p] /= num_batches
    model.train()
    return fisher

def run_ewc(num_epochs=2, lr=5e-5, ewc_lambda=50.0):
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)

    print("Estimating Fisher on French (protected) ...")
    fisher = estimate_fisher_on_french(model, num_batches=100)
    theta0 = copy.deepcopy(model).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    params = [p for p in model.parameters() if p.requires_grad]

    print("=== EWC: train on English with French EWC penalty ===")
    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            out = model(**batch)
            loss_new = out.loss

            ewc_loss = 0.0
            for p, p0 in zip(params, theta0.parameters()):
                ewc_loss = ewc_loss + (fisher[p] * (p - p0).pow(2)).sum()
            total_loss = loss_new + 0.5 * ewc_lambda * ewc_loss

            total_loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch} Step {step+1}] loss_new={loss_new.item():.4f}, ewc_loss={ewc_loss.item():.4f}")

        print(f"Epoch {epoch} evaluation:")
        eng_ppl = eval_ppl(model, english_test_ds, "English new (EWC)")
        fr_ppl  = eval_ppl(model, french_test_ds,  "French protected (EWC)")
    return model

ewc_model = run_ewc(num_epochs=3, lr=5e-5, ewc_lambda=50.0)


# %%
def run_protected_adam(
    num_epochs=2,
    lr=5e-5,
    alpha_geom=1.0,
    beta_geom=10.0,
    gamma_exp=0.5,
    subset_update_every=5,
    rho_all=0.99,
    rho_sub=0.99,
):
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]

    state = {}
    for p in params:
        state[p] = {
            "m": torch.zeros_like(p.data),
            "v_all": torch.zeros_like(p.data),
            "v_sub": torch.zeros_like(p.data),
        }

    beta1 = 0.9
    eps = 1e-6
    global_step = 0

    def protected_adam_step():
        nonlocal global_step
        global_step += 1
        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.data
            s = state[p]

            # first moment
            s["m"].mul_(beta1).add_(grad, alpha=1 - beta1)

            # second moment on "all" (new English) data
            s["v_all"].mul_(rho_all).addcmul_(grad, grad, value=1 - rho_all)

            v_all = s["v_all"]
            v_sub = s["v_sub"]
            v_protect = alpha_geom * v_all + beta_geom * v_sub

            m_hat = s["m"] / (1 - beta1**global_step)
            denom = (v_protect + eps).pow(gamma_exp)
            step = m_hat / denom
            p.data.add_(step, alpha=-lr)

    def update_subset_curvature():
        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.data
            s = state[p]
            s["v_sub"].mul_(rho_sub).addcmul_(grad, grad, value=1 - rho_sub)

    fr_iter = iter(fr_loader)

    print("=== ProtectedAdam-γ: geometry shaped by French subset ===")
    print(f"alpha_geom={alpha_geom}, beta_geom={beta_geom}, gamma_exp={gamma_exp}")
    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            # 1) English batch: gradient for new task
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()

            # 2) Take ProtectedAdam step (updates v_all + params)
            protected_adam_step()

            # 3) Occasionally update subset curvature using French
            if (step + 1) % subset_update_every == 0:
                try:
                    fr_batch = next(fr_iter)
                except StopIteration:
                    fr_iter = iter(fr_loader)
                    fr_batch = next(fr_iter)
                fr_batch = {k: v.to(device) for k, v in fr_batch.items()}
                model.zero_grad()
                fr_out = model(**fr_batch)
                fr_loss = fr_out.loss
                fr_loss.backward()
                update_subset_curvature()
                model.zero_grad()

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch} Step {step+1}] loss_new = {loss.item():.4f}")

        print(f"Epoch {epoch} evaluation:")
        eng_ppl = eval_ppl(model, english_test_ds, "English new (ProtectedAdam-γ)")
        fr_ppl  = eval_ppl(model, french_test_ds,  "French protected (ProtectedAdam-γ)")

    return model

protected_model = run_protected_adam(
    num_epochs=3,
    lr=1e-5,
    alpha_geom=1.0,
    beta_geom=10.0,   # strength of protected geometry
    gamma_exp=0.5,    # between 0.5 (Adam) and 1.0 (diag NGD)
    subset_update_every=5,
)


# %%
def run_protected_adam2(
    num_epochs=3,
    lr=5e-5,
    alpha_geom=1.0,
    beta_geom=10.0,
    gamma_exp=0.5,
    subset_update_every=5,
    rho_all=0.99,
    rho_sub=0.99,
):
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]

    state = {}
    for p in params:
        state[p] = {
            "m": torch.zeros_like(p.data),
            "v_all": torch.zeros_like(p.data),
            "v_sub": torch.zeros_like(p.data),
        }

    beta1 = 0.9
    eps = 1e-6
    global_step = 0

    def protected_adam_step():
        nonlocal global_step
        global_step += 1

        # First pass: update moments, compute v_protect, and accumulate
        # the mean denominators for γ=0.5 (baseline) and γ=gamma_exp
        temp = {}
        sum_baseline = 0.0
        sum_gamma = 0.0
        count_tensors = 0

        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.data
            s = state[p]

            # First moment
            s["m"].mul_(beta1).add_(grad, alpha=1 - beta1)

            # Second moment on "all" (new English) data
            s["v_all"].mul_(rho_all).addcmul_(grad, grad, value=1 - rho_all)

            v_all = s["v_all"]
            v_sub = s["v_sub"]
            v_protect = alpha_geom * v_all + beta_geom * v_sub

            # Bias-corrected first moment (optional but keeps Adam-like behaviour)
            m_hat = s["m"] / (1 - beta1**global_step)

            denom_baseline = (v_protect + eps).pow(0.5)
            denom_gamma = (v_protect + eps).pow(gamma_exp)

            sum_baseline += denom_baseline.mean()
            sum_gamma += denom_gamma.mean()
            count_tensors += 1

            temp[p] = {
                "m_hat": m_hat,
                "v_protect": v_protect,
            }

        if count_tensors == 0:
            return

        # Renormalization factor so that average step size matches γ=0.5 case
        scale = (sum_baseline / sum_gamma).detach()

        # Second pass: apply update with renormalized step size
        for p in params:
            if p.grad is None or p not in temp:
                continue
            buf = temp[p]
            m_hat = buf["m_hat"]
            v_protect = buf["v_protect"]

            denom_gamma = (v_protect + eps).pow(gamma_exp)
            step = (m_hat / denom_gamma) * scale
            p.data.add_(step, alpha=-lr)

    def update_subset_curvature():
        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.data
            s = state[p]
            s["v_sub"].mul_(rho_sub).addcmul_(grad, grad, value=1 - rho_sub)

    fr_iter = iter(fr_loader)

    print(f"Before opt:")
    eng_ppl = eval_ppl(model, english_test_ds, "English new (ProtectedAdam-γ)")
    fr_ppl  = eval_ppl(model, french_test_ds,  "French protected (ProtectedAdam-γ)")


    print("=== ProtectedAdam-γ: geometry shaped by French subset ===")
    print(f"alpha_geom={alpha_geom}, beta_geom={beta_geom}, gamma_exp={gamma_exp}")
    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            # 1) English batch: gradient for new task
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()

            # 2) Take ProtectedAdam step (updates v_all + params)
            protected_adam_step()

            # 3) Occasionally update subset curvature using French
            if (step + 1) % subset_update_every == 0:
                try:
                    fr_batch = next(fr_iter)
                except StopIteration:
                    fr_iter = iter(fr_loader)
                    fr_batch = next(fr_iter)
                fr_batch = {k: v.to(device) for k, v in fr_batch.items()}
                model.zero_grad()
                fr_out = model(**fr_batch)
                fr_loss = fr_out.loss
                fr_loss.backward()
                update_subset_curvature()
                model.zero_grad()

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch} Step {step+1}] loss_new = {loss.item():.4f}")

        print(f"Epoch {epoch} evaluation:")
        eng_ppl = eval_ppl(model, english_test_ds, "English new (ProtectedAdam-γ)")
        fr_ppl  = eval_ppl(model, french_test_ds,  "French protected (ProtectedAdam-γ)")

    return model


protected_model2 = run_protected_adam2(
    num_epochs=3,
    lr=5e-5,
    alpha_geom=1.0,
    beta_geom=10.0,   # strength of protected geometry
    gamma_exp=0.5,    # between 0.5 (Adam) and 1.0 (diag NGD)
    subset_update_every=5,
)


# %%


def run_replay(
    num_epochs=2,
    lr=5e-5,
    subset_update_every=5,
    replay_weight=1.0,   # λ: strength of French replay loss
):
    """
    Experience Replay baseline.

    - Optimizes English CE loss every step.
    - Every `subset_update_every` steps, also optimizes French CE.
    - Total loss = CE_english + replay_weight * CE_french.
    - Uses plain AdamW.
    - No curvature, no shielding, no geometry.
    """

    # Load model
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    # French iterator for replay
    fr_iter = iter(fr_loader)

    print("=== Replay baseline: English training + French replay ===")
    print(f"subset_update_every={subset_update_every}, replay_weight={replay_weight}")

    print("Before opt:")
    eval_ppl(model, english_test_ds, "English new (replay)")
    eval_ppl(model, french_test_ds,  "French protected (replay)")

    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # English forward/backward
            model.zero_grad()
            out = model(**batch)
            loss_new = out.loss
            total_loss = loss_new

            # French replay every N steps
            if (step + 1) % subset_update_every == 0:
                try:
                    fr_batch = next(fr_iter)
                except StopIteration:
                    fr_iter = iter(fr_loader)
                    fr_batch = next(fr_iter)
                fr_batch = {k: v.to(device) for k, v in fr_batch.items()}

                fr_out = model(**fr_batch)
                fr_loss = fr_out.loss

                total_loss = loss_new + replay_weight * fr_loss

            # Backprop + update
            total_loss.backward()
            optimizer.step()

            # Logging
            if (step + 1) % 100 == 0:
                if (step + 1) % subset_update_every == 0:
                    print(
                        f"[Epoch {epoch} Step {step+1}] "
                        f"loss_new={loss_new.item():.4f}, "
                        f"loss_replay={fr_loss.item():.4f}, "
                        f"total={total_loss.item():.4f}"
                    )
                else:
                    print(f"[Epoch {epoch} Step {step+1}] loss_new={loss_new.item():.4f}")

        # End epoch eval
        print(f"Epoch {epoch} evaluation:")
        eval_ppl(model, english_test_ds, "English new (replay)")
        eval_ppl(model, french_test_ds,  "French protected (replay)")

    return model


replay_model = run_replay(
    num_epochs=3,
    lr=5e-5,
    subset_update_every=5,
    replay_weight=1.0,
)


# %%
def estimate_fisher_french(model, num_batches=200):
    model.eval()
    fisher = {
        name: torch.zeros_like(p.data)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    loader = DataLoader(
        french_train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    it = iter(loader)

    for i in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()

        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            fisher[name] += p.grad.data.pow(2)

    for name in fisher:
        fisher[name] /= num_batches

    model.train()
    return fisher


# %%
def estimate_model_fisher_french(model, num_batches=200, top_k=100):
    """
    Compute *model Fisher* diagonal using KL(p_ref || p_model),
    with optional top-K truncation of the reference distribution.

    top_k < 0  → use full distribution (no truncation)
    top_k > 0  → keep only top_k tokens in reference distribution
    """

    # Freeze reference model θ0
    ref_model = copy.deepcopy(model).eval().to(device)
    for p in ref_model.parameters():
        p.requires_grad = False

    model.eval()

    fisher = {
        name: torch.zeros_like(p.data)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    loader = DataLoader(
        french_train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    it = iter(loader)

    for i in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: v.to(device) for k, v in batch.items()}

        # ---- 1. Reference distribution ----
        with torch.no_grad():
            ref_logits = ref_model(**batch).logits
            ref_probs_full = ref_logits.softmax(dim=-1)  # shape [B, T, V]

        # ---- 2. Possibly truncate to top-K ----
        if top_k is not None and top_k > 0:
            # Get top-K indices for each token
            top_vals, top_idx = torch.topk(ref_probs_full, k=top_k, dim=-1)
            # Renormalize probs over top-K
            ref_probs = top_vals / top_vals.sum(dim=-1, keepdim=True)
            # Make a tensor of zeros [B,T,V]
            ref_probs_k = torch.zeros_like(ref_probs_full)
            # Scatter top-K probabilities back into vocab dimension
            ref_probs_k.scatter_(-1, top_idx, ref_probs)
            ref_probs = ref_probs_k
        else:
            # use full distribution
            ref_probs = ref_probs_full

        # ---- 3. Model logits ----
        logits = model(**batch).logits
        log_probs = logits.log_softmax(dim=-1)

        # ---- 4. KL(p_ref || p_model) ----
        # KL per token: Σ_i q_i log(q_i/p_i)
        kl = (ref_probs * (ref_probs.log() - log_probs)).sum(dim=-1)
        loss = kl.mean()

        # ---- 5. Backprop = model Fisher at θ0 ----
        model.zero_grad()
        loss.backward()

        # ---- 6. Accumulate grad^2 ----
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            fisher[name] += p.grad.data.pow(2)

    # Average
    for name in fisher:
        fisher[name] /= num_batches

    model.train()
    return fisher


# %%
def run_protected_adam_precomputed(
    num_epochs=2,
    lr=5e-5,
    alpha_geom=1.0,
    beta_geom=10.0,
    gamma_exp=0.5,
    rho_all=0.99,
    fisher_sub=None,   # dict[name -> tensor]
):
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)
    model.train()

    # We'll work with named parameters for alignment
    named_params = [
        (name, p) for name, p in model.named_parameters()
        if p.requires_grad
    ]

    state = {}
    for name, p in named_params:
        if fisher_sub is not None and name in fisher_sub:
            v_sub_init = fisher_sub[name].clone().to(device)
        else:
            v_sub_init = torch.zeros_like(p.data)

        state[name] = {
            "m": torch.zeros_like(p.data),
            "v_all": torch.zeros_like(p.data),
            "v_sub": v_sub_init,
        }

    beta1 = 0.9
    eps = 1e-6
    global_step = 0

    def protected_adam_step():
        nonlocal global_step
        global_step += 1
        for name, p in named_params:
            if p.grad is None:
                continue
            grad = p.grad.data
            s = state[name]

            # first moment
            s["m"].mul_(beta1).add_(grad, alpha=1 - beta1)

            # second moment on "all" (new English) data
            s["v_all"].mul_(rho_all).addcmul_(grad, grad, value=1 - rho_all)

            v_all = s["v_all"]
            v_sub = s["v_sub"]  # fixed precomputed Fisher
            v_protect = alpha_geom * v_all + beta_geom * v_sub

            m_hat = s["m"] / (1 - beta1**global_step)
            denom = (v_protect + eps).pow(gamma_exp)
            step = m_hat / denom
            p.data.add_(step, alpha=-lr)

    print("=== ProtectedAdam-γ with precomputed French Fisher ===")
    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            protected_adam_step()

        print(f"Epoch {epoch} evaluation:")
        eval_ppl(model, english_test_ds, "English new (precomputed-Fisher)")
        eval_ppl(model, french_test_ds,  "French protected (precomputed-Fisher)")

    return model


# 1. Make a base model for Fisher estimation
base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
base_model.resize_token_embeddings(len(tokenizer))


# 2. Estimate Fisher on French ONCE
mfisher_french = estimate_model_fisher_french(base_model, num_batches=200)

# 3. Run English finetuning using precomputed Fisher, no French batches
protected_model_pre_mfisher = run_protected_adam_precomputed(
    num_epochs=3,
    lr=1e-5,
    alpha_geom=1.0,
    beta_geom=10.0,
    gamma_exp=0.5,
    rho_all=0.99,
    fisher_sub=mfisher_french,   # <- pass the dict here
)


# 2. Estimate Fisher on French ONCE
fisher_french = estimate_fisher_french(base_model, num_batches=200)

# 3. Run English finetuning using precomputed Fisher, no French batches
protected_model_pre = run_protected_adam_precomputed(
    num_epochs=3,
    lr=1e-5,
    alpha_geom=1.0,
    beta_geom=10.0,
    gamma_exp=0.5,
    rho_all=0.99,
    fisher_sub=fisher_french,   # <- pass the dict here
)


#

# %%
# does not work, ignore for now
def run_protected_adam_precomputed2(
    num_epochs=2,
    lr=5e-5,
    alpha_geom=1.0,      # scale for v_all (Adam geometry)
    beta_geom=10.0,      # strength of protection from v_sub
    gamma_exp=0.5,       # exponent applied only to normalized v_sub
    rho_all=0.99,
    fisher_sub=None,     # dict[name -> tensor], precomputed Fisher on French
):
    """
    Protected Adam with precomputed Fisher (additive version).

    - v_all: EMA of grad^2 on English (new task), like Adam.
    - v_sub: fixed Fisher from French (protected capability), precomputed.
    - v_sub is normalized globally once to be dimensionless.

    Update (per-parameter i):
        v_all_i ← EMA of g_i^2
        v_sub_i ≈ Fisher_i

        v_sub_scaled_i = v_sub_i / global_mean(v_sub)

        base_rms_i   = sqrt(alpha_geom * v_all_i)
        protect_i    = beta_geom * (v_sub_scaled_i ** gamma_exp)

        denom_i = base_rms_i + protect_i + eps

        Δθ_i = -lr * m_hat_i / denom_i

    Properties:
      - If fisher_sub is None or beta_geom = 0 -> exactly Adam.
      - If v_sub is small -> denom ≈ base_rms -> Adam-like.
      - If v_sub is large -> extra additive penalty in denom -> stronger protection.
    """

    # 1) Start from base GPT-2
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)
    model.train()

    # 2) Collect named parameters to align with fisher_sub[name]
    named_params = [
        (name, p) for name, p in model.named_parameters()
        if p.requires_grad
    ]

    # 3) Initialize state (m, v_all, v_sub)
    state = {}
    for name, p in named_params:
        v_sub_init = torch.zeros_like(p.data)
        if fisher_sub is not None and name in fisher_sub:
            v_sub_init = fisher_sub[name].clone().to(p.data.device)
        state[name] = {
            "m": torch.zeros_like(p.data),
            "v_all": torch.zeros_like(p.data),
            "v_sub": v_sub_init,
        }

    # 4) Compute a global mean of v_sub for normalization (dimensionless)
    if fisher_sub is not None:
        total_sum = 0.0
        total_count = 0
        for name, p in named_params:
            v_sub = state[name]["v_sub"]
            if v_sub.numel() > 0:
                total_sum += v_sub.sum().item()
                total_count += v_sub.numel()
        if total_count > 0:
            global_vsub_mean = total_sum / total_count
        else:
            global_vsub_mean = 1.0
    else:
        global_vsub_mean = 1.0

    beta1 = 0.9
    eps = 1e-8
    global_step = 0

    print("=== ProtectedAdam-precomputed2 (additive): Adam base + Fisher protection ===")
    print(
        f"alpha_geom={alpha_geom}, beta_geom={beta_geom}, "
        f"gamma_exp={gamma_exp}, rho_all={rho_all}, "
        f"global_vsub_mean={global_vsub_mean:.3e}"
    )

    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # ----- 1) Forward/backward on English (new task) -----
            model.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()

            global_step += 1

            # ----- 2) Protected Adam step (additive Fisher term) -----
            with torch.no_grad():
                for name, p in named_params:
                    if p.grad is None:
                        continue

                    g = p.grad.data
                    s = state[name]

                    # First moment (Adam)
                    s["m"].mul_(beta1).add_(g, alpha=1 - beta1)

                    # Second moment on "all" (new English) data (Adam-style)
                    s["v_all"].mul_(rho_all).addcmul_(g, g, value=1 - rho_all)

                    v_all = s["v_all"]
                    v_sub = s["v_sub"]

                    # Base Adam geometry: sqrt of v_all (scaled)
                    base_rms = (alpha_geom * v_all).sqrt()

                    # Normalized protective curvature from v_sub (dimensionless)
                    if fisher_sub is not None and beta_geom != 0.0 and global_vsub_mean > 0.0:
                        v_sub_scaled = v_sub / (global_vsub_mean + 1e-12)
                        v_sub_scaled = torch.clamp(v_sub_scaled, min=0.0)  # safety
                        protect_term = beta_geom * v_sub_scaled.pow(gamma_exp)
                    else:
                        protect_term = 0.0

                    m_hat = s["m"] / (1 - beta1**global_step)

                    # ADDITIVE protection: denom = base_rms + protective term
                    denom = base_rms + protect_term + eps
                    step_dir = m_hat / denom

                    p.data.add_(step_dir, alpha=-lr)

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch} Step {step+1}] loss_new = {loss.item():.4f}")

        # ----- 3) Epoch-end evaluation -----
        print(f"Epoch {epoch} evaluation:")
        eng_ppl = eval_ppl(model, english_test_ds, "English new (ProtAdam-pre2-add)")
        fr_ppl  = eval_ppl(model, french_test_ds,  "French protected (ProtAdam-pre2-add)")

    return model


# 3. Run English finetuning using precomputed Fisher, no French batches
protected_model_pre = run_protected_adam_precomputed2(
    num_epochs=3,
    lr=1e-5,
    alpha_geom=1.0,
    beta_geom=0.1,
    gamma_exp=0.5,
    rho_all=0.99,
    fisher_sub=fisher_french,   # <- pass the dict here
)



# %%
def run_adam_with_fisher_trust_region(
    num_epochs=2,
    lr=1e-5,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    fisher_sub=None,    # dict[name -> tensor] from estimate_fisher_french_named(...)
    delta_kl=1e-3,      # KL budget per step (approx)
):
    """
    Adam on English, with a TRPO-style KL trust region on French capability:
      1) Compute standard Adam step Δθ.
      2) Estimate French KL ≈ 0.5 * Σ_i F_sub[i] * (Δθ_i)^2
      3) If KL > delta_kl: scale Δθ by sqrt(delta_kl / KL).
    """

    # Start from the same base model as elsewhere
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)
    model.train()

    # Named params for alignment with fisher_sub
    named_params = [
        (name, p) for name, p in model.named_parameters()
        if p.requires_grad
    ]

    # Adam state
    state = {}
    for name, p in named_params:
        state[name] = {
            "m": torch.zeros_like(p.data),
            "v": torch.zeros_like(p.data),
        }

    global_step = 0

    print("=== Adam with Fisher KL trust region on French ===")
    print(f"lr={lr}, delta_kl={delta_kl}")
    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # 1) Forward/backward on English batch
            model.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()

            global_step += 1

            # 2) Compute Adam proposal step Δθ for each param (WITHOUT applying yet)
            proposed_steps = {}  # name -> tensor (Δθ)
            for name, p in named_params:
                if p.grad is None:
                    proposed_steps[name] = torch.zeros_like(p.data)
                    continue

                g = p.grad.data
                s = state[name]

                # Adam moments
                s["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                s["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Bias-corrected
                m_hat = s["m"] / (1 - beta1 ** global_step)
                v_hat = s["v"] / (1 - beta2 ** global_step)

                # Classic Adam step (note: step is *direction*, no lr yet)
                step_dir = m_hat / (v_hat.sqrt() + eps)

                # Proposed parameter change Δθ = -lr * step_dir
                delta_theta = -lr * step_dir
                proposed_steps[name] = delta_theta

            # 3) Estimate French KL for this joint step using precomputed Fisher
            kl_est = 0.0
            if fisher_sub is not None:
                for name, p in named_params:
                    if name not in fisher_sub:
                        continue
                    delta = proposed_steps[name]
                    if delta is None:
                        continue
                    F = fisher_sub[name].to(delta.device)
                    # 0.5 * sum_i F_i * (Δθ_i)^2
                    kl_est += 0.5 * (F * (delta ** 2)).sum().item()

            # 4) Compute scaling factor to enforce KL ≤ delta_kl
            if fisher_sub is None or kl_est <= 0.0:
                scale = 1.0
            elif kl_est <= delta_kl:
                scale = 1.0
            else:
                scale = (delta_kl / kl_est) ** 0.5

            # 5) Apply scaled step
            for name, p in named_params:
                delta = proposed_steps[name]
                if delta is None:
                    continue
                p.data.add_(delta * scale)

            if (step + 1) % 100 == 0:
                print(
                    f"[Epoch {epoch} Step {step+1}] "
                    f"loss_new = {loss.item():.4f}, KL_est = {kl_est:.3e}, scale = {scale:.3f}"
                )

        # 6) Evaluation at epoch end
        print(f"Epoch {epoch} evaluation:")
        eng_ppl = eval_ppl(model, english_test_ds, "English new (Adam+KL)")
        fr_ppl  = eval_ppl(model, french_test_ds,  "French protected (Adam+KL)")

    return model




# Precompute model Fisher on French (once)
base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
base_model.resize_token_embeddings(len(tokenizer))

fisher_french = estimate_fisher_french(base_model, num_batches=200)

# Now run English finetuning with TRPO-style KL trust region on French
adam_trpo_model = run_adam_with_fisher_trust_region(
    num_epochs=3,
    lr=1e-5,
    fisher_sub=fisher_french,
    delta_kl=1e-10,   # tune this up/down
)


# %%
print("=== Final comparison ===")
print("Baseline Adam:")
eval_ppl(baseline_model, english_test_ds, "English new (baseline)")
eval_ppl(baseline_model, french_test_ds,  "French protected (baseline)")

print("\nEWC:")
eval_ppl(ewc_model, english_test_ds, "English new (EWC)")
eval_ppl(ewc_model, french_test_ds,  "French protected (EWC)")

print("\nProtectedAdam-γ:")
eval_ppl(protected_model, english_test_ds, "English new (ProtectedAdam-γ)")
eval_ppl(protected_model, french_test_ds,  "French protected (ProtectedAdam-γ)")


print("\nProtectedAdam2-γ:")
eval_ppl(protected_model2, english_test_ds, "English new (ProtectedAdam-γ)")
eval_ppl(protected_model2, french_test_ds,  "French protected (ProtectedAdam-γ)")

print("\nReplay:")
eval_ppl(replay_model, english_test_ds, "English new (ProtectedAdam-γ)")
eval_ppl(replay_model, french_test_ds,  "French protected (ProtectedAdam-γ)")




