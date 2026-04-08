# Network Data

**Theme:** Network Data: Graph Construction + Measurement

## Goals
- Construct graphs from relational data.
- Compute core measures (centrality, communities).
- See how modeling choices change conclusions.

## Materials
- `demo/` — in-class demo code (`networks.py`: SBM generation, centrality, Louvain, GCN)
- `data/raw/` — datasets (see `data/README.md` for full documentation)
  - **cosponsorship_nodes.csv** / **cosponsorship_edges.csv** — primary dataset (500-node synthetic legislative cosponsorship network with party labels and 6 node features)
  - **karate_edges.csv** / **karate_nodes.csv** — Zachary's Karate Club (34 nodes, quick demos)
  - **les_miserables_edges.csv** / **les_miserables_nodes.csv** — Les Misérables co-appearances (77 nodes)
- `data/download_real_data.py` — optional script to fetch polbooks + Cora on student machines
- `slides/` — lecture slides and slide-generation code
- `problem_set/` — LaTeX problem set (`network_data.tex`)

## Readings
- Minhas, S & P. Hoff (2025) Political Analysis
- Olivella et al. (2022) JAS
