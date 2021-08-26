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

run locally
~~~
#make sure you got trained models in state_dict folder
cd text_summary_endpoint
sh build_and_push.sh
docker run -v -d -p 8080:8080 bertseq2seq
~~~

```python
# test 
#curl http://localhost:8080/ping 

# curl
import requests
import json

url='http://localhost:8080/invocations'
data={"data": "《半導體》Q1展望保守，世界垂淚2019/02/11 10:28時報資訊【時報記者沈培華台北報導】世界先進 (5347) 去年營運創歷史新高，每股純益達3.72元。但對今年首季展望保守，預計營收將比上季高點減近一成。世界先進於封關前股價拉高，今早則是開平走低。世界先進於年前台股封關後舉行法說會公布財報。公司去年營運表現亮麗，營收與獲利同創歷史新高紀錄。2018年全年營收289.28億元，年增16.1%，毛利率35.2%，拉升3.2個百分點，稅後淨利61.66億元，年增36.9%，營收與獲利同創歷史新高，每股純益3.72元。董事會通過去年度擬配發現金股利3.2元。展望第一季，受到客戶進入庫存調整，公司預期，本季營收估在67億至71億元，將季減8%至13%，毛利率將約34.5%至36.5%。此外，因應客戶需求，世界先進決定斥資2.36億美元，收購格芯新加坡8吋晶圓廠。世界先進於年前宣布，將購買格芯位於新加坡Tampines的8吋晶圓3E廠房、廠務設施、機器設備及微機電(MEMS)智財權與業務，交易總金額2.36億美元，交割日訂108年12月31日。格芯晶圓3E廠現有月產能3.5萬片8吋晶圓，世界先進每年將可增加超過40萬片8吋晶圓產能，增進公司明年起業績成長動能。TOP關閉"}
data = json.dumps(data)
r = requests.post(url,data=data)

#show result
print (r.text)
```

结果如下
```json
{"摘要": "2011 年 f1 周 年 展 望 保 守 保 守 世 界 第 一 (图 )"}
```

run on endpoint

```shell script
endpoint_ecr_image="251885400447.dkr.ecr.cn-northwest-1.amazonaws.com.cn/bertseq2seq"
python create_endpoint.py \
--endpoint_ecr_image_path ${endpoint_ecr_image} \
--endpoint_name 'bertseq2seq' \
--instance_type "ml.g4dn.xlarge"
```

在部署结束后，看到SageMaker控制台生成了对应的endpoint,可以使用如下客户端代码测试调用
```python
from boto3.session import Session
import json
data={"data": "《半導體》Q1展望保守，世界垂淚2019/02/11 10:28時報資訊【時報記者沈培華台北報導】世界先進 (5347) 去年營運創歷史新高，每股純益達3.72元。但對今年首季展望保守，預計營收將比上季高點減近一成。世界先進於封關前股價拉高，今早則是開平走低。世界先進於年前台股封關後舉行法說會公布財報。公司去年營運表現亮麗，營收與獲利同創歷史新高紀錄。2018年全年營收289.28億元，年增16.1%，毛利率35.2%，拉升3.2個百分點，稅後淨利61.66億元，年增36.9%，營收與獲利同創歷史新高，每股純益3.72元。董事會通過去年度擬配發現金股利3.2元。展望第一季，受到客戶進入庫存調整，公司預期，本季營收估在67億至71億元，將季減8%至13%，毛利率將約34.5%至36.5%。此外，因應客戶需求，世界先進決定斥資2.36億美元，收購格芯新加坡8吋晶圓廠。世界先進於年前宣布，將購買格芯位於新加坡Tampines的8吋晶圓3E廠房、廠務設施、機器設備及微機電(MEMS)智財權與業務，交易總金額2.36億美元，交割日訂108年12月31日。格芯晶圓3E廠現有月產能3.5萬片8吋晶圓，世界先進每年將可增加超過40萬片8吋晶圓產能，增進公司明年起業績成長動能。TOP關閉"}

session = Session()
    
runtime = session.client("runtime.sagemaker")
response = runtime.invoke_endpoint(
    EndpointName='bertseq2seq',
    ContentType="application/json",
    Body=json.dumps(data),
)

result = json.loads(response["Body"].read())
print (result)
```



