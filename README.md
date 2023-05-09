# ESNR

This repository includes the ESNR (Edge signal-to-noise ratio) part for paper "Towards Understanding and Reducing Graph Structural Noise for GNNs". ESNR is proposed as a novel metric for measuring graph structural noise level for real graph-structured datasets.

## Dependencies

    numpy
    torch==1.13.0
    sklearn
    torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    torch-geometric==2.2.0
