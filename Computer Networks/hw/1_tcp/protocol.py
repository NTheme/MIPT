import socket
import time
from queue import PriorityQueue


class TCPSegment:
    SERVICE_LEN = 8
    ACK_TIMEOUT = 0.01

    def __init__(self, seq_number: int, ack_number: int, data: bytes):
        self.seq_number = seq_number
        self.ack_number = ack_number
        self.data = data
        self.acknowledged = False
        self._sending_time = time.time()

    def dump(self) -> bytes:
        seq = self.seq_number.to_bytes(TCPSegment.SERVICE_LEN, "big", signed=False)
        ack = self.ack_number.to_bytes(TCPSegment.SERVICE_LEN, "big", signed=False)
        return seq + ack + self.data

    def update_sending_time(self, sending_time=None):
        self._sending_time = sending_time if sending_time is not None else time.time()

    @staticmethod
    def load(data: bytes) -> 'TCPSegment':
        seq = int.from_bytes(data[:TCPSegment.SERVICE_LEN], "big", signed=False)
        ack = int.from_bytes(data[TCPSegment.SERVICE_LEN:2 * TCPSegment.SERVICE_LEN], "big", signed=False)
        return TCPSegment(seq, ack, data[2 * TCPSegment.SERVICE_LEN:])

    @property
    def expired(self):
        return not self.acknowledged and (time.time() - self._sending_time > self.ACK_TIMEOUT)

    def __len__(self):
        return len(self.data)

    def __lt__(self, other):
        return self.seq_number < other.seq_number

    def __eq__(self, other):
        return self.seq_number == other.seq_number


class UDPBasedProtocol:
    def __init__(self, *, local_addr, remote_addr):
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.remote_addr = remote_addr
        self.udp_socket.bind(local_addr)

    def sendto(self, data):
        return self.udp_socket.sendto(data, self.remote_addr)

    def recvfrom(self, n):
        msg, addr = self.udp_socket.recvfrom(n)
        return msg

    def close(self):
        self.udp_socket.close()


class MyTCPProtocol(UDPBasedProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mss = 2 ** 10
        self.window_size = 2 ** 12
        self.ack_crit_lag = 2 ** 5

        self._sent_bytes_n = 0
        self._confirmed_bytes_n = 0
        self._received_bytes_n = 0
        self._send_window = PriorityQueue()
        self._recv_window = PriorityQueue()
        self._buffer = bytes()

    def recv(self, n: int) -> bytes:
        right_border = min(n, len(self._buffer))
        data = self._buffer[:right_border]
        self._buffer = self._buffer[right_border:]
        while len(data) < n:
            self._receive_segment()
            right_border = min(n, len(self._buffer))
            data += self._buffer[:right_border]
            self._buffer = self._buffer[right_border:]
        return data

    def _receive_segment(self, timeout: float = None) -> bool:
        self.udp_socket.settimeout(timeout)
        try:
            segment = TCPSegment.load(self.recvfrom(self.mss + 2 * TCPSegment.SERVICE_LEN))
        except socket.error:
            return False
        if len(segment):
            self._recv_window.put((segment.seq_number, segment), block=False)
            self._shift_recv_window()
        if segment.ack_number > self._confirmed_bytes_n:
            self._confirmed_bytes_n = segment.ack_number
            self._shift_send_window()
        return True

    def send(self, data: bytes) -> int:
        sent_data_len = 0
        lag = 0
        while (lag < self.ack_crit_lag) and (data or self._confirmed_bytes_n < self._sent_bytes_n):
            if (self._sent_bytes_n - self._confirmed_bytes_n <= self.window_size) and data:
                right_border = min(self.mss, len(data))
                sent_length = self._send_segment(TCPSegment(self._sent_bytes_n,
                                                            self._received_bytes_n,
                                                            data[: right_border]))
                data = data[sent_length:]
                sent_data_len += sent_length
            else:
                if self._receive_segment(TCPSegment.ACK_TIMEOUT):
                    lag = 0
                else:
                    lag += 1
            self._resend_first_segment()
        return sent_data_len

    def _send_segment(self, segment: TCPSegment) -> int:
        self.udp_socket.settimeout(None)
        sent_length = self.sendto(segment.dump()) - 2 * segment.SERVICE_LEN

        if segment.seq_number == self._sent_bytes_n:
            self._sent_bytes_n += sent_length
        elif segment.seq_number > self._sent_bytes_n:
            raise ValueError()
        if len(segment):
            segment.data = segment.data[: sent_length]
            segment.update_sending_time()
            self._send_window.put((segment.seq_number, segment), block=False)
        return sent_length

    def _shift_recv_window(self):
        first_segment = None
        while not self._recv_window.empty():
            _, first_segment = self._recv_window.get(block=False)
            if first_segment.seq_number < self._received_bytes_n:
                first_segment.acknowledged = True
            elif first_segment.seq_number == self._received_bytes_n:
                self._buffer += first_segment.data
                self._received_bytes_n += len(first_segment)
                first_segment.acknowledged = True
            else:
                self._recv_window.put((first_segment.seq_number, first_segment), block=False)
                break
        if first_segment is not None:
            self._send_segment(TCPSegment(self._sent_bytes_n, self._received_bytes_n, bytes()))

    def _shift_send_window(self):
        while not self._send_window.empty():
            _, first_segment = self._send_window.get(block=False)
            if first_segment.seq_number >= self._confirmed_bytes_n:
                self._send_window.put((first_segment.seq_number, first_segment), block=False)
                break

    def _resend_first_segment(self, force=False):
        if self._send_window.empty():
            return
        _, first_segment = self._send_window.get(block=False)
        if first_segment.expired or force:
            self._send_segment(first_segment)
        else:
            self._send_window.put((first_segment.seq_number, first_segment), block=False)

    def close(self):
        super().close()
