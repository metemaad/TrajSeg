import numpy as np
import matplotlib.pyplot as plt


class Cluster:
    cluster_id = 0
    eps = 0
    min_points = 0
    max_points = 0
    trajectory_points = None
    def cluster_stat(self):
        s={}
        for tp in self.trajectory_points:
            if tp.label not in s.keys():
                s[tp.label]=1
            else:
                s[tp.label]=s[tp.label]+1
        return s
    def cluster_stat_mode(self):
        s={}
        for tp in self.trajectory_points:
            if tp.transportation_mode not in s.keys():
                s[tp.transportation_mode]=1
            else:
                s[tp.transportation_mode]=s[tp.transportation_mode]+1
        return s

    def get_traj_ids(self):
        traj_id=[]
        for tp in self.trajectory_points:
            traj_id.append(tp.trajectory_id)
        return traj_id
    def plot(self,clr='r',marker_='o'):
        for tp in self.trajectory_points:
            plt.scatter(tp.latitude,tp.longitude,c=clr,marker=marker_)


    def check_min_points(self):
        n = len(self.Trajectory_Points)
        if n < self.min_points:
            return False
        else:
            return True

    def get_cluster_id(self):
        return self.cluster_id

    def get_cardinality(self):
        if self.trajectory_points is None:
            return 0
        else:
            return len(self.trajectory_points)

    def reset(self):
        self.trajectory_points = None

    def assign_point(self, tp):
        if self.trajectory_points is None:
            self.trajectory_points = np.array([tp])
            print(" add to cluster ",str(self.cluster_id))
        else:
            self.trajectory_points = np.append(self.trajectory_points, [tp])
            print(" assign to cluster ", str(self.cluster_id))

    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        # print("new cluster")


"""
    def calculate_eps(self, window_buffer):
        self.window_buffer = window_buffer
        # print("Calculate eps for ", self.window_buffer)
        self.eps = self.window_buffer.calculate_features()
        print("eps:", self.eps)
        # c = window_buffer.cluster(self.eps, self.cluster_id)
        # find location of -1
        print(self.eps)
        return self.eps

        pass
"""
