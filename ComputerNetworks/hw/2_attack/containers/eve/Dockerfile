FROM ubuntu:22.04

RUN apt update && apt install -y curl iproute2 dnsutils inetutils-ping net-tools iptables ncat \
    git make gcc tcpdump vim zsh python3-pip python3-dev  \
    libnfnetlink-dev libnetfilter-queue-dev build-essential libnetfilter-queue-dev

WORKDIR /root/attack
RUN pip install netfilterqueue scapy
RUN git clone https://github.com/smikims/arpspoof.git arpspoof-src && cd arpspoof-src && make && cp arpspoof ..
COPY ./attack.py .
COPY ./attack.sh .
RUN chmod +x attack.sh
