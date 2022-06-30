FROM nvcr.io/nvidia/pytorch:21.10-py3
    
# Install some basic utilities and python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Install requirements first
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mislight ./mislight
COPY parameters ./parameters

COPY predict.sh .
