# ESNR

This repository includes the ESNR (Edge signal-to-noise ratio) part for paper "Towards Understanding and Reducing Graph Structural Noise for GNNs". ESNR is proposed as a novel metric for measuring graph structural noise level for real graph-structured datasets.

<img width="1160" alt="image" src="https://github.com/MingzeDong/ESNR/assets/68533876/7c91b717-a51f-4d96-bcf7-fab8a4f160a2">



## Dependencies

    numpy
    torch==1.13.0
    sklearn
    torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    torch-geometric==2.2.0

## Results

ESNR performance in synthetic contextual stochastic block model:
<img width="1287" alt="image" src="https://github.com/MingzeDong/ESNR/assets/68533876/2d8f020c-ac83-4a17-8804-a4c3fe6cb593">


ESNR performance in real graph-structured datasets:

<img width="869" alt="image" src="https://github.com/MingzeDong/ESNR/assets/68533876/4b3a4127-14c1-4df4-8b4a-ae49f5ff1437">


For more details, please refer to our paper: [Towards Understanding and Reducing Graph Structural Noise for GNNs](https://proceedings.mlr.press/v202/dong23a.html)
```
@InProceedings{pmlr-v202-dong23a,
  title = 	 {Towards Understanding and Reducing Graph Structural Noise for {GNN}s},
  author =       {Dong, Mingze and Kluger, Yuval},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {8202--8226},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR}
}
```
