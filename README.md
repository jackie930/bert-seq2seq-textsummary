# bert-seq2seq-textsummary

使用bert seq2seq进行金融新闻文档摘要

## train

首先下载预训练模型文件，`bert_seq2seq-textsummary/state_dict/`,目录下存放`bert_config.json`,`pytorch_model.bin`,`vocab.txt`


下载pretrain-models
~~~
cd bert-seq2seq-textsummary
mkdir state_dict && cd state_dict
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz && tar -zxvf bert-base-chinese.tar.gz
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
~~~

下载THUNews
~~~
cd ..
mkdir data
mkdir data/sports
mkdir data/finance
aws s3 cp s3://datalab2021-cn/textsummary/data/THUCNews/体育/ data/sports/ --recursive
aws s3 cp s3://datalab2021-cn/textsummary/data/THUCNews/财经/ data/finance/ --recursive
~~~

训练model-baseline
~~~
source activate pytorch_p36
pip install -r requirements.txt
#单卡训练
python examples/summary.py
#多卡训练
python examples/summary-multi.py
~~~

datalab模型finetune（补充小样本csv/excel数据）
~~~
#训练数据准备
python utils/process.py
#训练
python examples/summary.py
~~~

## predict
python test/auto_title_test.py 
## deploy