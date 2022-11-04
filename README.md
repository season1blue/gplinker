# ccf2022-spo-baseline
[高端装备制造知识图谱自动化构建技术评测任务](https://www.datafountain.cn/competitions/584/datasets)

### 赛题任务

​	通过从大量故障案例文本抽取出部件单元、性能表征、故障状态、检测工具等实体及其关系，可以为后续高端装备制造业故障知识图谱构建和故障智能检修和实时诊断打下坚实基础。本任务需要从故障案例文本自动抽取4种类型的关系和4种类型的实体。关系类型为：部件单元的故障状态、性能表征的故障状态、部件单元和性能表征的检测工具、部件单元之间的组成关系。

### baseline

​     使用苏神的spo联合抽取baseline，对数据生成方式进行了改造，目前baseline线上 F1>=0.63

### 运行

首先pip添加依赖 requirements.txt

进入train目录运行baseline.py

相关内容：[GPLinker：基于GlobalPointer的实体关系联合抽取](https://kexue.fm/archives/8888)

### 环境

Python 3.8 + Keras 2.3.1 + Tensorflow-gpu 2.2.0 + bert4keras 0.11.3

# gplinker
