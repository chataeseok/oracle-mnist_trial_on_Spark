# Oracle-MNIST trials on Spark

[![Readme-EN](https://img.shields.io/badge/README-English-purple.svg)](README.md)

これは、`Spark`分散計算フレームワークを使用した`Oracle-MNIST`データセットに基づく2つの拡張性のテストです。試験では、畳み込みニューラルネットワーク(`LeNet-5`。`CNN.py`を参照)と畳み込み生成対抗ネットワーク(`CGAN.py`を参照)を訓練した。ノートパソコン`test_after_training_CGAN.ipynb`は生成効果を試験した。

## 原始論文:
A dataset of oracle characters for benchmarking machine learning algorithms. Mei Wang, Weihong Deng. 
[Scientific Data](https://www.nature.com/articles/s41597-024-02933-w)

## 元のリポジトリ:
[https://github.com/wm-bupt/oracle-mnist](https://github.com/wm-bupt/oracle-mnist)

## CNN.py の用法:
```bash
spark-submit CNN.py \
# カスタムリソース割り当て,たとえば、
--master <あなたのSpark standalone url> \
--total-executor-cores 4 \
--executor-cores 1 \
--executor-memory 2G \
--driver-memory 4G \
```

## CGAN.py の用法
```bash
spark-submit CGAN.py \
# カスタムリソース割り当て,たとえば、
--master <your Spark standalone url> \
--total-executor-cores 4 \
--executor-cores 1 \
--executor-memory 2G \
--driver-memory 4G \
```

## test_after_training_CGAN.ipynb の用法:
ノート内のセルを実行する。

ここに原作者への心からの感謝を申し上げます。