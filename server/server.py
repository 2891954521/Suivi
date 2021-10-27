import socket
import threading

import cv2
import numpy as np
from numpy.core.fromnumeric import clip

from recognition import RecognitionServer

# 压缩参数，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]

class Client(threading.Thread):
    
    def __init__(self, image: RecognitionServer, connect):
        super().__init__()
        self.image = image
        self.connect = connect

    def run(self):
        while True:
            try:
                with self.image.condition:
                    self.image.condition.wait()
                    _, imgencode = cv2.imencode('.jpg', self.image.frame, encode_param)
                    # 将numpy矩阵转换成字符形式，以便在网络中传输
                    stringData = np.array(imgencode).tostring()
                    # 先发送要发送的数据的长度
                    # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
                    self.connect.send(str.encode(str(len(stringData)).ljust(16)))
                    # 发送数据
                    self.connect.send(stringData)
            except (ConnectionResetError, BrokenPipeError) as e:
                print('断开连接！')
                break

        self.connect.close()


class SocketServer(threading.Thread):

    def __init__(self, image: RecognitionServer):
        super().__init__()

        self.image = image

        self.connect = None

        self.isFinish = False

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('0.0.0.0', 8081))
        # 参数指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
        self.server.listen(3)

    def run(self):

        while True:

            if self.isFinish:
                break

            connect, address = self.server.accept()

            print('connect from:' + str(address))

            client = Client(self.image, connect)

            client.start()



