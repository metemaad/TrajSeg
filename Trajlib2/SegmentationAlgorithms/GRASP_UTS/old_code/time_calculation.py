import time


class Clock():

    def __init__(self):
        self.t1 = time.time()
        self.last_time = 0
        self.name = ""
        self.average = 0
        self.count = 0
        self.min = 20000
        self.max = 0
    def start(self, name):
        self.t1 = time.time()
        self.name = name
    def stop(self):
        out = time.time() - self.t1
        self.last_time = out
        print(self.name + ": %f" % (out))
        self.average += out
        self.count += 1
        if out < self.min:
            self.min = out
        if out > self.max:
            self.max = out
    def get_avg(self):
        return round(self.average / self.count,2)
