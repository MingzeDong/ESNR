# ESNR

This repository includes the ESNR (Edge signal-to-noise ratio) part for paper "Towards Understanding and Reducing Graph Structural Noise for GNNs". ESNR is proposed as a novel metric for measuring graph structural noise level for real graph-structured datasets.

<img width="1328" alt="image" src="https://github.com/MingzeDong/ESNR/assets/68533876/c4cb29c8-3bc3-4868-b42f-11d069b083e1">


## Dependencies

    numpy
    torch==1.13.0
    sklearn
    torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    torch-geometric==2.2.0

## Results

ESNR performance in synthetic contextual stochastic block model:
<img width="1448" alt="image" src="https://github.com/MingzeDong/ESNR/assets/68533876/7b8a6eaf-80a9-4a98-887d-f63c2b82930a">

ESNR performance in real graph-structured datasets:

<img width="700" alt="image" src="https://github.com/MingzeDong/ESNR/assets/68533876/e820e8db-99fb-4b2e-b872-076c9693f5d1">

For more details, please refer to our paper:
