from server import SocketServer
from recognition import RecognitionServer

if __name__ == '__main__':

    recognition = RecognitionServer()

    server = SocketServer(recognition)
    
    recognition.start()
    server.start()

    server.join()

    print("退出主线程")

    exit()
