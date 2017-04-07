import os

class Log(object):
    def __init__(self):
        self.msgPool = []

    def add(self, msg):
        self.msgPool.append(msg)

    def outputLog(self, path):
        with open(path, 'wb') as fid:
            for line in self.msgPool:
                line += '\n'
                fid.write(bytes(line, 'UTF-8'))

