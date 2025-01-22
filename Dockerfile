FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN conda update -n base -c defaults conda && \
    conda install -y python=3.12 && \
    conda update --all --yes

COPY ./requirements.txt /install/requirements.txt
COPY ./requirements-dev.txt /install/requirements-dev.txt
COPY ./requirements-thirdparty.txt /install/requirements-thirdparty.txt

RUN pip install -r /install/requirements.txt
RUN pip install -r /install/requirements-dev.txt
# RUN pip install -r /install/requirements-thirdparty.txt
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
RUN pip install typing_extensions>=4.4.0
ENV PYTHONPATH="src/:thirdparty/VPS"
ENV TZ="Asia/Singapore"

WORKDIR /code
