
# pip install pyaudio websockets
import pyaudio
from threading import Thread
from websockets.sync.client import connect
import time

p = pyaudio.PyAudio()
speak = p.open(format=pyaudio.paInt16, channels=1, rate=32000, output=True)

with connect("ws://localhost:9880/ws") as websocket:
    def rec(websocket):
        while True:
            recv_text = websocket.recv()
            speak.write(recv_text)
    Thread(target=rec, args=(websocket,)).start()
    while True:
        rrs = input("请输入你的问题：")
        if rrs:
            websocket.send(rrs)
            time.sleep(10)
