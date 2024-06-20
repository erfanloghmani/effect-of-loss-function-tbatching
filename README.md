# Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks

This repository contains the code for the research paper titled "Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks." The code is based on the [JODIE](https://github.com/claws-lab/jodie) project, and alternative loss functions have been added to investigate their impact on dynamic network representation learning.

## Getting Started

To run the code with different loss functions, you need to specify the desired model using command-line arguments.

- For the original loss function, use `--model jodie`.
- For $loss_{item-sum}$, use `--model jodie-sum`.
- For $loss_{full-sum}$, use `--model jodie-full-sum`.

Both the `jodie.py` script for training and the `evaluate_interaction_prediction.py` script for validation and testing accept these command-line arguments.

## Datasets

This paper introduces a new dataset related to android application install interactions in the Myket android application market. The dataset can be accessed from the following link: [Myket Android Application Install Dataset](https://github.com/erfanloghmani/myket-android-application-market-dataset/).

In addition to the new dataset, we also use three other datasets from the [JODIE](https://github.com/claws-lab/jodie) project:

- Reddit
- LastFM
- Wikipedia

The dataset `.csv` files should be placed in the `data/` directory. For the Myket dataset for instance, you should put the [myket.csv](https://raw.githubusercontent.com/erfanloghmani/myket-android-application-market-dataset/main/myket.csv) file under the path `data/myket.csv`.

## Citation

If you use this code in your research or work, please cite the following [preprint](https://arxiv.org/abs/2308.06862):

```
@article{loghmani2023effect,
  author       = {Erfan Loghmani and MohammadAmin Fazli},
  title        = {Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks},
  journal      = {CoRR},
  volume       = {abs/2308.06862},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2308.06862},
  doi          = {10.48550/ARXIV.2308.06862},
  eprinttype    = {arXiv},
  eprint       = {2308.06862},
  timestamp    = {Wed, 23 Aug 2023 14:43:32 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2308-06862.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
