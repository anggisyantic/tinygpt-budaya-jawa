import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_blocks import Block

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# ==========================
# Load Corpus
# ==========================

with open("budayajawa.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Word-level Tokenizer

words = text.split()

vocab = sorted(list(set(words)))

stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

data = torch.tensor(
    [stoi[w] for w in words],
    dtype=torch.long
)

vocab_size = len(vocab)

print("Vocabulary Size:", vocab_size)

# ==========================
# Hyperparameter
# ==========================

block_size = 16
embedding_dim = 64
n_heads = 2
n_layers = 2

lr = 1e-3
epochs = 3000

# ==========================
# Batch Generator
# ==========================

def get_batch(batch_size=16):

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack(
        [data[i:i+block_size] for i in ix]
    )

    y = torch.stack(
        [data[i+1:i+block_size+1] for i in ix]
    )

    return x, y

# ==========================
# TinyGPT Model
# ==========================

class TinyGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    embedding_dim,
                    block_size,
                    n_heads
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(
            embedding_dim
        )

        self.head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(
            torch.arange(
                T,
                device=idx.device
            )
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.view(B*T, C),
                targets.view(B*T)
            )

        return logits, loss

    def generate(
        self,
        idx,
        max_new_tokens
    ):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_idx = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_idx),
                dim=1
            )

        return idx

# ==========================
# Training
# ==========================

model = TinyGPT()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)

for step in range(epochs):

    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 300 == 0:
        print(
            f"Step {step}, loss={loss.item():.4f}"
        )

# ==========================
# Save Model
# ==========================

torch.save(
    model.state_dict(),
    "tinygpt_word_model.pth"
)

print("Model saved!")

# ==========================
# Text Generation
# ==========================

prompt = "Budaya Jawa merupakan"

prompt_words = prompt.split()

context = torch.tensor(
    [[stoi[w] for w in prompt_words if w in stoi]],
    dtype=torch.long
)

out = model.generate(
    context,
    max_new_tokens=50
)

generated_ids = out[0].tolist()

generated_words = [
    itos[i]
    for i in generated_ids
]

generated_text = " ".join(
    generated_words
)

print("\nGenerated Text:\n")
print(generated_text)