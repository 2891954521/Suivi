import sys
import socket

import cv2
import numpy


# 接受TCP的数据，count为接收的数据量
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


try:
    IP = '127.0.0.1'#    '192.168.5.2'
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((IP, 8081))
except socket.error as msg:
    print(msg)
    sys.exit(1)

while True:

    length = recvall(client, 16)

    stringData = recvall(client, int(length))

    # 将获取到的字符流数据转换成1维数组
    data = numpy.frombuffer(stringData, numpy.uint8)
    # 将数组解码成图像
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    cv2.imshow('SERVER', image)

    if cv2.waitKey(10) & 0xff == 27:
        break

client.close()

cv2.destroyAllWindows()
