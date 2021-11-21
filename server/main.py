from server import SocketServer
from recognition import RecognitionServer

if __name__ == '__main__':

    recognition = RecognitionServer()

    server = SocketServer(recognition)
    
    recognition.start()
    server.start()

    try:

        while True:
            pass

    except KeyboardInterrupt:
        print("退出主线程")
        recognition.isFinish = True
        server.isFinish = True

    exit()
