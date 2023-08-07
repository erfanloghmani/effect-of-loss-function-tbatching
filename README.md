# Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks

This repository contains the code for the research paper titled "Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks." The code is based on the [JODIE](https://github.com/claws-lab/jodie) project, and alternative loss functions have been added to investigate their impact on dynamic network representation learning.

## Getting Started

To run the code with different loss functions, you need to specify the desired model using command-line arguments.

- For the original loss function, use `--model jodie`.
- For $loss_{item-sum}$, use `--model jodie-sum`.
- For $loss_{full-sum}$, use `--model jodie-full-sum`.

Both the `jodie.py` script for training and the `evaluate_interaction_prediction.py` script for validation and testing accept these command-line arguments.

## Citation

If you use this code in your research or work, please cite the following paper:

```

```
