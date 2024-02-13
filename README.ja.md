# Oracle-MNIST trial on Spark

[![Readme-EN](https://img.shields.io/badge/README-English-purple.svg)](README.md)

`Spark` 分散計算フレームワークを使用した `Oracle MNIST` データセットに基づく拡張可能な試験です。この実験では、 `LeNet-5` ニューラルネットワークを訓練しました。

原始論文:
A dataset of oracle characters for benchmarking machine learning algorithms. Mei Wang, Weihong Deng. 
[Scientific Data](https://www.nature.com/articles/s41597-024-02933-w)

元のリポジトリ:
[https://github.com/wm-bupt/oracle-mnist](https://github.com/wm-bupt/oracle-mnist)

用法:
```bash
spark-submit training.py \
--master <ユーザのSpark standalone url> \
...(カスタムリソース割り当て)
```

ここに原作者への心からの感謝を申し上げます。