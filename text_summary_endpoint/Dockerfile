ARG REGISTRY_URI
FROM ${REGISTRY_URI}/pytorch-inference:1.5.0-gpu-py36-cu101-ubuntu16.04

ENV PATH="/opt/ml/code:${PATH}"
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-unauthenticated \
    jq

RUN ldconfig -v

RUN apt-get install ffmpeg libsm6 libxext6 -y

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

RUN pip install --upgrade pip
RUN pip install flask gevent gunicorn boto3 jieba

WORKDIR /opt/ml/code
COPY bert_seq2seq/. /opt/ml/code/bert_seq2seq/
COPY state_dict/. /opt/ml/code/state_dict/

COPY serve.py /opt/ml/code
COPY wsgi.py /opt/ml/code
COPY predictor.py /opt/ml/code
COPY nginx.conf /opt/ml/code


ENTRYPOINT ["python","serve.py"]