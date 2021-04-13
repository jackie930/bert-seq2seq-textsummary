# bert-seq2seq-textsummary

使用bert seq2seq进行金融新闻文档摘要

## train

首先下载预训练模型文件，`bert_seq2seq-textsummary/state_dict/`,目录下存放`bert_config.json`,`pytorch_model.bin`,`vocab.txt`

然后准备训练数据，运行下面的脚本进行模型训练。
~~~
source activate pytorch_p36
#训练数据准备
python utils/process.py
#训练
python examples/summary.py
~~~

## predict
python test/auto_title_test.py 
## deploy