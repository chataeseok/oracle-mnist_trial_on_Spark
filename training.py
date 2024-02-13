"""
This is a distributed training script
分散型トレーニングスクリプトです

DATA SOURCE
データソース:
            https://github.com/wm-bupt/oracle-mnist

USAGE
用法:
            spark-submit training.py \
            --master <user's Spark standalone url ユーザのSpark standalone url> \
            ...(custom source assignment カスタムリソース割り当て)

COMPILED BY
コンパイラ:
            Chung Wong
            キカイチュウ

REFERRED TO
リファレンス:
            wm-bupt's https://github.com/wm-bupt/oracle-mnist/blob/main/src/mnist_reader.py
            Andrew Gu's https://github.com/pytorch/examples/blob/1aa2eec9ac94102ac479cd88396b8aa3f2429092/distributed/ddp/example.py
"""

import sys
sys.path.append('./')
import torch
import torch.nn as nn
from torchvision import transforms
import gzip
from typing_extensions import Literal
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Oracle_mnist').getOrCreate()

"""
Parameters pre-setting
事前のパラメータ設定
"""
classes_n = 10#number of classes
batch_size = 256#batch size
learning_rate = 0.01#learning rate
epochs_n = 100#number of training epochs
scheduler_step = 10#period of learning rate decay
gamma = 0.95#multiplicative factor of learning rate decay

'''Network
ニューラルネットワーク'''
class LeNet_5(nn.Module):
    """
    Modification of LeNet-5
    LeNet-5ネットワークのバリエーション
    """
    def __init__(self):
        """
        Initialization
        初期化
        """
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
        self.norm1 = nn.BatchNorm2d(6)
        self.norm2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.LeakyReLU()
        self.linear1 = nn.Linear(256, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output_linear = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input: torch.Tensor):#[batch_size, 1, 28, 28]
        """
        Forward propagation
        フォワード伝播
        """
        output = self.activation(self.norm1(self.conv1(input)))#[batch_size, 6, 24, 24],24=(28-5)/1+1
        output = self.dropout(self.pool(output))#[batch_size, 6, 12, 12],12=(24-2)/2+1
        output = self.activation(self.norm2(self.conv2(output)))#[batch_size, 16, 8, 8],8=(12-5)/1+1
        output = self.dropout(self.pool(output))#[batch_size, 16, 4, 4],4=(8-2)/2+1
        output = output.reshape(-1, 256)#[batch_size, 256],256=16X4X4

        output = self.dropout(self.activation(self.linear1(output)))#1st Fully connected layer[batch_size, 120]
        output = self.dropout(self.activation(self.linear2(output)))#2nd Fully connected layer[batch_size, 84]
        output = self.output_linear(output)#output layer[batch_size, 10]

        return output

    def init_model(self):
        """
        Model initialization
        モデル初期化
        """
        for name, layer in self._modules.items():
            if name == 'conv1':
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_in', nonlinearity = 'relu')
                nn.init.constant_(layer.bias, 0)
            elif name == 'conv2':
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std = 0.01)
                nn.init.normal_(layer.bias, std = 0.01)

    def loadparam(self, path: str):
        """
        Load pre-saved model parameters
        ロード前に保存したモデルパラメータ
        """
        self.load_state_dict(torch.load(path, map_location = torch.device('cpu')))

    def savemodel(self, path: str = './network.pth'):
        """
        Save the trained model
        訓練されたモデルを保存する
        """
        torch.save(self.state_dict(), path)

    def showparam(self):
        """
        Show model parameters
        モデルパラメータの表示
        """
        return [(name, param) for name, param in self.named_parameters()]

"""
Functions for loading datasets
データセットをロードする関数
"""
def loaddata(img_trans, type: Literal['train', 't10k'] = 'train'):
    """
    Import Oracle-mnist dataset from binary buffer
    バイナリ・バッファからOracle-mnistデータセットをインポートするには
    """
    with gzip.open(f'./data/oracle/{type}-labels-idx1-ubyte.gz', 'rb') as lb_path:
        lbs = torch.frombuffer(lb_path.read(), dtype = torch.uint8, offset = 8).clone().long()

    with gzip.open(f'./data/oracle/{type}-images-idx3-ubyte.gz', 'rb') as img_path:
        imgs = torch.frombuffer(img_path.read(), dtype = torch.uint8, offset = 16).clone().reshape(len(lbs), 28, 28).float()

    dataset = [(img_trans(img), lb) for img, lb in zip(imgs, lbs)]
    return dataset

"""
Import data and preprocess data
データのインポートと前処理データ
"""
# ori_trainset = loaddata(img_trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()))
# train_std1, train_mean1 = torch.std_mean(torch.stack([img for img, _ in ori_trainset]))#the standard deviation and mean of channel pixels in the original images
# print(f'std:{train_std1:.4f},mean:{train_mean1:.4f}')#0.3713, 0.3814

norm_trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.3814,), (0.3713,))])#standardize images

trainset = loaddata(img_trans = norm_trans)

def train(epochs_n: int = 100, learning_rate: float = 0.01, scheduler_step: int = 10, gamma: float = 0.95):
    """
    distributed training
    分散トレーニング
    """
    import torch
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torch.utils.data import DataLoader, DistributedSampler
    import torchmetrics as tm
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    cost_log, acc_log = torch.tensor([]), torch.tensor([])#cost history and accuracy history during initialization training

    dist.init_process_group('gloo')#distributed environment initialization

    ori_model = LeNet_5()#instantiate model
    ori_model.init_model()#initialize model
    model = DDP(ori_model)#parallelize model

    distributed_sampler = DistributedSampler(trainset)#distributed slices of samples
    train_loader = DataLoader(dataset = trainset, batch_size = batch_size, sampler = distributed_sampler, shuffle = False)

    cost_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = scheduler_step, gamma = gamma)

    model.train()
    for epoch in range(1, epochs_n + 1):
        acc = tm.Accuracy(task = 'multiclass', num_classes = classes_n)#Initialize training accuracy
        losses = 0.#Initialize training losses
        for x, y in train_loader:            
            y_pred = model(x)
            cost = cost_fn(y_pred, y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            acc.update(y_pred, y)
            losses += cost.detach() * y.shape[0]
        
        scheduler.step()

        acc_log = torch.cat([acc_log, acc.compute().reshape(-1)])
        cost_log = torch.cat([cost_log, (losses / len(trainset)).reshape(-1)])

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}, training cost:{cost.item():.4f}, training accuracy:{acc.compute().item():.2%}')

    dist.destroy_process_group()#cleaning up distributed environments
    model.module.savemodel()#saving network weights parameters and it should be the default options to save the main process(rank=0)
    torch.save(acc_log, './acc_log.pth')
    torch.save(cost_log, './cost_log.pth')

"""
distributed execution
分散計算の実行
"""
distributor = TorchDistributor(num_processes = 2, #number of processes
                            local_mode = False, #Using local mode or computing clusters
                            use_gpu = False)
print('-'*20 + 'start training' + '-'*20)
distributor.run(train, epochs_n, learning_rate, scheduler_step, gamma)#Users can adjust the training parameters according to the situation
print('-'*20 + 'training done' + '-'*20)

spark.stop()