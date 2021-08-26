# bert-seq2seq-textsummary

使用`bert seq2seq`基于THUNews进行中文文档摘要（生成式），并在`AWS SageMaker`上进行部署

# model

# 中文数据集THUCNews
THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。
每个新闻均为txt文件，第一句话为新闻摘要。

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
#全量thunews后台训练
nohup python -u examples/summary-multi.py > train.log 2>&1 &
~~~

## local predict
~~~
#使用训练好的模型预测单个句子
python test/auto_title_test.py 
~~~

## deploy on AWS SageMaker Endpoint

~~~
#make sure you got trained models in state_dict folder
cd text_summary_endpoint
sh build_and_push.sh
docker run -v -d -p 8080:8080 bertseq2seq

~~~

