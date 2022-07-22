FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    emacs \
    git \
    less \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    tree \
    unzip \
    vim \
    wget \
    xterm \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
COPY requirements.txt /root
RUN pip3 install -r /root/requirements.txt

COPY *.py /root/
COPY *.sh /root/
COPY modeldir/ /root/modeldir/

RUN chmod a+x /root/train.sh && chmod a+x /root/test.sh
ENV PATH $PATH:/root/

WORKDIR /root/
