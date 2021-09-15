FROM hub-dev.hexin.cn/jupyterhub/nvidia_cuda:py37-cuda100-ubuntu18.04-v2

COPY ./ /home/jovyan/reward-order-longtext-extraction 

RUN cd /home/jovyan/reward-order-longtext-extraction  && \
    python -m pip install -r requirements.txt 
