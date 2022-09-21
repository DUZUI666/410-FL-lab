## What is this

本项目基于https://github.com/shaoxiongji/federated-learning， 对项目中的各个部分进行了解耦，以方便大家新加需求，进行实验



## Requirements

Python >= 3.8

pytorch >= 1.12.1

CUDA >= 11.3



## Introduction to some parts

### models

#### Client

主要包括Client类，每一个Client都有一个单独的DataLoader，在主函数里通过Indice对数据集进行划分，再通过Client中的DatasetSplit类构造出



#### Server

主要包括Server类，可以在类内定义新的聚合算法



#### Nets

用于训练的模型，可以在此定义新的模型用于训练



### utils

#### DataTest

封装了用于进行模型评估的函数

#### EnvInit

封装了实验环境初始化的函数，如果需要特殊的实验环境，可在此扩展

#### options

包括实验的参数，可以在此调整。加了新需求也可以在这里列出新的args条目。需要注意args贯彻了整个实验，如Client，Server的构造函数等

#### sampling

数据集划分使用的函数

### main_FL

联邦学习的主函数，默认有最基础的FL_train()。可以在其中定义其他的训练方法







