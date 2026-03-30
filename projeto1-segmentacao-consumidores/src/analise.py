"""
Segmentação de Consumidores por Comportamento de Compra
========================================================
Análise exploratória + clusterização com K-Means
Dados sintéticos inspirados no perfil do consumidor brasileiro (POF/IBGE)

Autor: Andrade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Configuração visual ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#2e3250",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
})

PALETTE = ["#00b4d8", "#f77f00", "#06d6a0", "#e63946", "#a8dadc"]

# ── 1. Geração de dados sintéticos ───────────────────────────────────────────
np.random.seed(42)
N = 800

def gera_consumidores(n):
    """
    Simula perfis de consumidores brasileiros com base em faixas de renda
    e padrões de consumo típicos (inspirado na POF 2017-2018).
    """
    renda = np.concatenate([
        np.random.normal(1500,  300,  int(n * 0.35)),   # Classe C/D
        np.random.normal(4000,  800,  int(n * 0.40)),   # Classe C/B
        np.random.normal(10000, 2000, int(n * 0.25)),   # Classe B/A
    ])
    renda = np.clip(renda, 800, 30000)

    # Gasto mensal como proporção da renda (com ruído)
    gasto_alimentacao    = renda * np.random.uniform(0.20, 0.40, n) + np.random.normal(0, 100, n)
    gasto_lazer          = renda * np.random.uniform(0.05, 0.20, n) + np.random.normal(0, 80, n)
    gasto_vestuario      = renda * np.random.uniform(0.04, 0.12, n) + np.random.normal(0, 50, n)
    gasto_digital        = renda * np.random.uniform(0.02, 0.08, n) + np.random.normal(0, 30, n)
    frequencia_compras   = np.random.poisson(lam=renda / 1000 + 2, size=n)
    ticket_medio         = (gasto_alimentacao + gasto_lazer) / (frequencia_compras + 1)

    df = pd.DataFrame({
        "renda_mensal":       np.abs(renda),
        "gasto_alimentacao":  np.abs(gasto_alimentacao),
        "gasto_lazer":        np.abs(gasto_lazer),
        "gasto_vestuario":    np.abs(gasto_vestuario),
        "gasto_digital":      np.abs(gasto_digital),
        "freq_compras_mes":   frequencia_compras,
        "ticket_medio":       np.abs(ticket_medio),
    })
    return df

df = gera_consumidores(N)
df.to_csv("data/consumidores_raw.csv", index=False)
print(f"Dataset gerado: {df.shape[0]} consumidores, {df.shape[1]} variáveis")
print(df.describe().round(2))

# ── 2. EDA ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribuição das Variáveis de Consumo", fontsize=16, fontweight="bold", y=1.02)

variaveis = ["renda_mensal", "gasto_alimentacao", "gasto_lazer",
             "gasto_vestuario", "gasto_digital", "ticket_medio"]
labels    = ["Renda Mensal (R$)", "Gasto Alimentação (R$)", "Gasto Lazer (R$)",
             "Gasto Vestuário (R$)", "Gasto Digital (R$)", "Ticket Médio (R$)"]

for ax, var, label, cor in zip(axes.flat, variaveis, labels, PALETTE * 2):
    ax.hist(df[var], bins=40, color=cor, alpha=0.85, edgecolor="none")
    ax.set_title(label, fontsize=11, pad=8)
    ax.set_xlabel("Valor (R$)")
    ax.set_ylabel("Frequência")
    media = df[var].mean()
    ax.axvline(media, color="#ffffff", linestyle="--", linewidth=1.2, label=f"Média: R${media:,.0f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/01_distribuicoes.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfico 1 salvo: distribuições")

# Heatmap de correlação
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", ax=ax,
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    linewidths=0.5, linecolor="#2e3250",
    annot_kws={"size": 9},
)
ax.set_title("Correlação entre Variáveis de Consumo", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/02_correlacao.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfico 2 salvo: correlação")

# ── 3. Clusterização K-Means ─────────────────────────────────────────────────
features = ["renda_mensal", "gasto_lazer", "gasto_digital", "freq_compras_mes", "ticket_medio"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do cotovelo + silhouette
inertias, silhouettes = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels_k))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Seleção do Número Ideal de Clusters", fontsize=14, fontweight="bold")

ax1.plot(K_range, inertias, "o-", color=PALETTE[0], linewidth=2, markersize=7)
ax1.set_title("Método do Cotovelo (Inércia)")
ax1.set_xlabel("Número de Clusters (k)")
ax1.set_ylabel("Inércia")
ax1.axvline(4, color=PALETTE[1], linestyle="--", linewidth=1.5, label="k=4 escolhido")
ax1.legend()

ax2.plot(K_range, silhouettes, "s-", color=PALETTE[2], linewidth=2, markersize=7)
ax2.set_title("Coeficiente de Silhouette")
ax2.set_xlabel("Número de Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.axvline(4, color=PALETTE[1], linestyle="--", linewidth=1.5, label="k=4 escolhido")
ax2.legend()

plt.tight_layout()
plt.savefig("outputs/03_selecao_k.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfico 3 salvo: seleção de k")

# Modelo final com k=4
K_FINAL = 4
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_scaled)

# ── 4. Visualização dos clusters via PCA ─────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["pca1"] = coords[:, 0]
df["pca2"] = coords[:, 1]

var_exp = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(12, 8))
nomes_clusters = {0: "Econômico", 1: "Aspiracional", 2: "Premium", 3: "Digital-first"}

for c, (nome, cor) in enumerate(zip(nomes_clusters.values(), PALETTE)):
    mask = df["cluster"] == c
    ax.scatter(df.loc[mask, "pca1"], df.loc[mask, "pca2"],
               c=cor, alpha=0.65, s=45, label=f"Cluster {c}: {nome}", edgecolors="none")

ax.set_title("Segmentação de Consumidores — Projeção PCA", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variância explicada)")
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variância explicada)")
ax.legend(fontsize=11, framealpha=0.2)
plt.tight_layout()
plt.savefig("outputs/04_clusters_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfico 4 salvo: clusters PCA")

# ── 5. Perfil de cada cluster ────────────────────────────────────────────────
perfil = df.groupby("cluster")[features + ["gasto_alimentacao", "gasto_vestuario"]].mean().round(2)
perfil["nome"] = perfil.index.map(nomes_clusters)
print("\n── Perfil médio por cluster ──────────────────────────────")
print(perfil.to_string())
perfil.to_csv("outputs/perfil_clusters.csv")

fig, ax = plt.subplots(figsize=(13, 6))
categorias = ["renda_mensal", "gasto_alimentacao", "gasto_lazer",
              "gasto_vestuario", "gasto_digital", "ticket_medio"]
x = np.arange(len(categorias))
width = 0.2

for i, (c, nome, cor) in enumerate(zip(range(K_FINAL), nomes_clusters.values(), PALETTE)):
    vals = [perfil.loc[c, col] for col in categorias]
    ax.bar(x + i * width, vals, width, label=f"Cluster {c}: {nome}", color=cor, alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(["Renda", "Alimentação", "Lazer", "Vestuário", "Digital", "Ticket Médio"],
                   fontsize=10)
ax.set_title("Perfil Médio por Segmento de Consumidor", fontsize=14, fontweight="bold")
ax.set_ylabel("Valor Médio (R$)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/05_perfil_clusters.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✓ Gráfico 5 salvo: perfil dos clusters")
print(f"\n✅ Análise concluída. {K_FINAL} segmentos identificados.")
print("   Outputs salvos em /outputs/")
