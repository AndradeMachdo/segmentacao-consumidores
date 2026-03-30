# 🛒 Segmentação de Consumidores com K-Means

> Análise exploratória e clusterização de perfis de consumo inspirados na realidade do consumidor brasileiro.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-completo-brightgreen)

---

## 📌 Objetivo

Identificar **segmentos de consumidores** com base em padrões de renda e comportamento de gasto, usando análise exploratória e algoritmo K-Means. O projeto simula dados inspirados na **Pesquisa de Orçamentos Familiares (POF/IBGE)** e entrega perfis acionáveis para estratégias de marketing.

---

## 🧠 Metodologia

```
Dados Sintéticos (POF-inspired)
        ↓
Análise Exploratória (EDA)
        ↓
Padronização (StandardScaler)
        ↓
Seleção de k (Elbow + Silhouette)
        ↓
K-Means Clustering (k=4)
        ↓
Visualização via PCA 2D
        ↓
Perfis por Segmento
```

---

## 📊 Segmentos Identificados

| Cluster | Nome | Renda Média | Comportamento |
|---------|------|-------------|---------------|
| 0 | **Econômico** | ~R$ 1.500 | Alta proporção do gasto em alimentação; baixo gasto digital |
| 1 | **Aspiracional** | ~R$ 4.000 | Equilíbrio entre categorias; frequência de compra moderada |
| 2 | **Premium** | ~R$ 10.000+ | Alto ticket médio, lazer e vestuário elevados |
| 3 | **Digital-first** | ~R$ 5.000 | Gasto digital desproporcional; alta frequência de compras online |

---

## 🗂️ Estrutura do Projeto

```
projeto1-segmentacao-consumidores/
│
├── data/
│   └── consumidores_raw.csv        # Dataset gerado
│
├── src/
│   └── analise.py                  # Script principal
│
├── outputs/
│   ├── 01_distribuicoes.png        # Histogramas das variáveis
│   ├── 02_correlacao.png           # Heatmap de correlação
│   ├── 03_selecao_k.png            # Elbow + Silhouette
│   ├── 04_clusters_pca.png         # Projeção 2D dos clusters
│   ├── 05_perfil_clusters.png      # Barras comparativas por segmento
│   └── perfil_clusters.csv         # Médias por cluster
│
└── README.md
```

---

## ▶️ Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/segmentacao-consumidores.git
cd segmentacao-consumidores

# 2. Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Execute a análise
python src/analise.py
```

---

## 📈 Principais Resultados

- **Silhouette Score com k=4: ~0.38** — segmentação estatisticamente defensável
- A variável mais discriminante entre clusters é a **renda mensal**, seguida de **gasto digital**
- O segmento *Digital-first* representa ~20% da base e tem potencial de crescimento acelerado
- Clusters visualizados em 2D via PCA explicam **~68% da variância total**

---

## 🛠️ Tecnologias

- `pandas` / `numpy` — manipulação de dados
- `scikit-learn` — K-Means, PCA, StandardScaler, Silhouette Score
- `matplotlib` / `seaborn` — visualizações

---

## 👤 Autor

**Andrade** · Estatística — UERJ  
Marketing Analytics & Consumer Behavior  
📍 São Gonçalo, Rio de Janeiro
