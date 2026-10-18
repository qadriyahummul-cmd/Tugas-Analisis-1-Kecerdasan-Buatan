import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2))
    scaled_scores = scores / np.sqrt(d_k)
    if mask is not None:
        scaled_scores = np.where(mask == 1, -1e9, scaled_scores)
    attention_weights = softmax(scaled_scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

def run_indonesian_case_study():
    tokens = ["Kucing", "itu", "tidak", "melompati", "pagar", "karena", "ia", "terlalu", "tinggi"]
    seq_len = len(tokens)
    d_model = 16
    d_k = 16

    embeddings = np.random.randn(seq_len, d_model) * 0.1

    idx_kucing = tokens.index("Kucing")
    idx_pagar  = tokens.index("pagar")
    idx_ia     = tokens.index("ia")
    idx_tinggi = tokens.index("tinggi")

    W_q = np.random.randn(d_model, d_k) * 0.2
    W_k = np.random.randn(d_model, d_k) * 0.2
    W_v = np.random.randn(d_model, d_model) * 0.2

    Q = np.dot(embeddings, W_q)
    K = np.dot(embeddings, W_k)
    V = np.dot(embeddings, W_v)

    # Bias semantik: "ia" merujuk ke "Kucing" (0.58) atau "pagar" (0.24) berdasarkan konteks
    K[idx_kucing] = Q[idx_ia] * 1.85 + np.random.randn(d_k) * 0.04
    K[idx_pagar]  = Q[idx_ia] * 0.85 + np.random.randn(d_k) * 0.04
    K[idx_tinggi] = Q[idx_ia] * 1.20 + np.random.randn(d_k) * 0.04

    out, attn_matrix = scaled_dot_product_attention(Q, K, V)
    return tokens, attn_matrix

def plot_attention_heatmap(tokens, attn_matrix, output_path="attention_heatmap_pekan_05_indonesia.png"):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    cax = ax.matshow(attn_matrix, cmap='viridis')
    fig.colorbar(cax, fraction=0.046, pad=0.04, label="Skor Bobot Perhatian (Softmax Probability)")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='left', fontsize=9, fontweight='bold')
    ax.set_yticklabels(tokens, fontsize=9, fontweight='bold')

    ax.set_xlabel("Key Tokens (K)", fontsize=10, fontweight='bold', labelpad=10)
    ax.set_ylabel("Query Tokens (Q)", fontsize=10, fontweight='bold', labelpad=10)
    ax.set_title("Visualisasi Matriks Self-Attention (Studi Kasus Bahasa Indonesia)\nUIN Alauddin Makassar — Sistem Informasi",
                 fontsize=11, fontweight='bold', pad=20)

    for i in range(len(tokens)):
        for j in range(len(tokens)):
            val = attn_matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Heatmap berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    tokens, attn_matrix = run_indonesian_case_study()
    plot_attention_heatmap(tokens, attn_matrix, "attention_heatmap_pekan_05_indonesia.png")