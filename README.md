# Oracle-MNIST trials on Spark

[![Readme-JA](https://img.shields.io/badge/README-Japanese-red.svg)](README.ja.md)

These are 2 expandable trials based on the `Oracle-MNIST` dataset, using the `Spark` distributed computing framework. In trials, a convolutional neural network(`LeNet-5`, see `CNN.py`) and a convolutional generative adversarial network(see `CGAN.py`) were trained. And notebook `test_after_training_CGAN.ipynb` tested the generation effect.

## Original paper:
A dataset of oracle characters for benchmarking machine learning algorithms. Mei Wang, Weihong Deng. 
[Scientific Data](https://www.nature.com/articles/s41597-024-02933-w)

## Original repository: 
[https://github.com/wm-bupt/oracle-mnist](https://github.com/wm-bupt/oracle-mnist)

## CNN.py usage:
```bash
spark-submit CNN.py \
# custom source assignment, for example,
--master <your Spark standalone url> \
--total-executor-cores 4 \
--executor-cores 1 \
--executor-memory 2G \
--driver-memory 4G \
```

## CGAN.py usage:
```bash
spark-submit CGAN.py \
# custom source assignment, for example,
--master <your Spark standalone url> \
--total-executor-cores 4 \
--executor-cores 1 \
--executor-memory 2G \
--driver-memory 4G \
```

## test_after_training_CGAN.ipynb usage:
After training the network, run cells in notebook.

##
I would like to express my sincere gratitude to the original authors.