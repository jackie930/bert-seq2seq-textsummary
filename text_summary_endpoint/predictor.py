import warnings
import torch
import sys
sys.path.append("./")
sys.path.append("./bert_seq2seq")
import json
warnings.filterwarnings("ignore",category=FutureWarning)

from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert

import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

import flask

# The flask app for serving predictions
app = flask.Flask(__name__)

#init loading
auto_title_model = "./state_dict/bert_auto_title_model-sports.bin"
vocab_path = "./state_dict/bert-base-chinese-vocab.txt"  # 模型字典的位置

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert"  # 选择模型名字
# model_path = "./state_dict/bert-base-chinese-pytorch_model.bin"  # 模型
# 加载字典
word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)
# 定义模型
bert_model = load_bert(word2idx, model_name=model_name)
bert_model.set_device(device)
bert_model.eval()
## 加载训练的模型参数
bert_model.load_all_params(model_path=auto_title_model, device=device)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")
    data = flask.request.data.decode('utf-8')
    print ("<<<<<input data: ", data)
    print ("<<<<<input content type: ", flask.request.content_type)

    data = json.loads(data)
    data_input = data['data']

    print('Invoked with {} records'.format(data.keys()))
    with torch.no_grad():
        res = bert_model.generate(data_input, beam_size=3)

    result = {"摘要":res}
    print ("<<<result: ", result)

    response = json.dumps(result,ensure_ascii=False)

    return flask.Response(response=response, status=200, mimetype='application/json')