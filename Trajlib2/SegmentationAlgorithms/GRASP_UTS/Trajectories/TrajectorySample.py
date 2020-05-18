class TrajectorySample:
    def __init__(self):
        self.tid = 0
        self.firstGid = 0
        self.lastGid = 0
        self.samplePoints = {}
        self.weights = []

    def copy(self):
        s = TrajectorySample()
        s.tid = self.tid
        s.firstGid = self.firstGid
        s.lastGid = self.lastGid
        s.samplePoints = dict(self.samplePoints)
        s.weights = list(self.samplePoints)
        return s
    def __str__(self):
        return str(self.tid)
    def __repr__(self):
        return str(self.tid)
