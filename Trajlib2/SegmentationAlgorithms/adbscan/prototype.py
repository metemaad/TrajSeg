from Adaptive_DBSCAN.cluster import Cluster
from Adaptive_DBSCAN.segment import Segment
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


class ADBSCAN:
    cluster_id = 0
    eps = 0
    sensitivity=3
    min_points = 3
    max_points = 0
    clusters = None
    speed_condition = True
    bearing_cond = True
    current_cluster = None
    noise_cluster = None
    current_trajectory_point = None  # the point before window buffer
    window_size = 10
    number_of_processed_points=0
    full_traj=[]

    window_buffer = None  # a segment
    Status = "idle"  # idle clustering wait

    def all_eps(self):
        colors = ['r','b','g','y','k','#225ea8', '#41b6c4', '#a1dab4', '#ffffcc']

        print("noise :(", self.noise_cluster.get_cardinality(), ")")


        if self.noise_cluster is not None:
            if self.noise_cluster.trajectory_points is not None:
                self.noise_cluster.plot(clr='r',marker_='*')
                print(self.noise_cluster.cluster_stat())
                print(self.noise_cluster.cluster_stat_mode())
        i=0
        for c in self.clusters:
            print(c.cluster_id, ":", c.eps, "(", c.get_cardinality(), ")")
            c.plot(clr=colors[i])
            print(c.cluster_stat())
            print(c.cluster_stat_mode())
            i=(i+1)%len(colors)
        plt.show()
    def get_segment_id(self):
        segment_id=[-1]*self.number_of_processed_points


        #if self.noise_cluster is not None:
        #    if self.noise_cluster.trajectory_points is not None:
        #        self.noise_cluster.plot(clr='r',marker_='*')
        #        print(self.noise_cluster.cluster_stat())
        #        print(self.noise_cluster.cluster_stat_mode())

        for c in self.clusters:
            traj_ids=c.get_traj_ids()
            for t in traj_ids:
                segment_id[t]=c.cluster_id
        return segment_id



    def add_new_point(self, trajectory_point):
        pass

    def wait_for_window(self, tp):
        if self.window_buffer is None:
            # create window buffer and wait
            self.window_buffer = Segment(tp,sensitivity=self.sensitivity,speed_condition=self.speed_condition,
                                         bearing_cond=self.bearing_cond)
            self.Status = "wait"
        else:
            if self.window_buffer.get_duration_of_segment() >= self.window_size:
                print("window is ready for c=", str(self.current_cluster.cluster_id))
                self.Status = "clustering"
                self.cluster_trajectory_point(tp)
                self.current_cluster.eps = self.window_buffer.get_all_eps()
                # eps = self.current_cluster.calculate_eps(self.window_buffer)
            else:
                # buffer is not full. wait for more data
                self.window_buffer.add_point(tp)
                pass

    def add_point_to_cluster(self, tp):
        # plt.scatter(tp.latitude, tp.longitude, c='g')

        if self.current_cluster.trajectory_points is None:
            self.current_cluster.trajectory_points = np.array([tp])
            print("add to cluster ", str(self.current_cluster.cluster_id), " [", str(tp.latitude), ",",
                  str(tp.longitude) + "]")
        else:
            self.current_cluster.trajectory_points = np.append(self.current_cluster.trajectory_points, np.array([tp]))
            print("add to cluster ", str(self.current_cluster.cluster_id), " [", str(tp.latitude), ",",
                  str(tp.longitude) + "]")

    def plot_steps(self,tp):
        plt.figure()
        x_min =  tp.latitude
        x_max =  tp.latitude
        y_min =  tp.longitude
        y_max =  tp.longitude

        # plot nise
        if self.noise_cluster is not None:
            if self.noise_cluster.trajectory_points is not None:
                for tp0 in self.noise_cluster.trajectory_points:
                    plt.scatter(tp0.latitude, tp0.longitude, c='r',marker='*')
                    x_min=np.fmin(x_min,tp0.latitude)
                    x_max = np.fmax(x_max, tp0.latitude)
                    y_min=np.fmin(y_min,tp0.longitude)
                    y_max = np.fmax(y_max, tp0.longitude)




        if self.window_buffer is not None:
            i=0
            for tp0 in self.window_buffer.trajectory_points:
                print("****  buffer",str(i),":[",str(tp0.latitude),str(tp0.longitude)+"]",str(tp0.timeDate))
                plt.scatter(tp0.latitude,tp0.longitude,c='r')
                plt.annotate("b"+str(i)+"|", (tp0.latitude,tp0.longitude))
                x_min = np.fmin(x_min, tp0.latitude)
                x_max = np.fmax(x_max, tp0.latitude)
                y_min = np.fmin(y_min, tp0.longitude)
                y_max = np.fmax(y_max, tp0.longitude)
                i=i+1
        plt.scatter(tp.latitude,tp.longitude,c='b')
        plt.annotate("next " +str(tp.timeDate), (tp.latitude, tp.longitude))
        print("next point:", str(tp.latitude), str(tp.longitude),str(tp.timeDate))
        if self.current_trajectory_point is not None:
            plt.scatter(self.current_trajectory_point.latitude, self.current_trajectory_point.longitude, c='g')
            plt.annotate("curr " +str(self.current_trajectory_point.timeDate), (self.current_trajectory_point.latitude, self.current_trajectory_point.longitude))
            print("Curr point:", str(self.current_trajectory_point.latitude), str(self.current_trajectory_point.longitude),str(self.current_trajectory_point.timeDate))
            x_min = np.fmin(x_min, self.current_trajectory_point.latitude)
            x_max = np.fmax(x_max, self.current_trajectory_point.latitude)
            y_min = np.fmin(y_min, self.current_trajectory_point.longitude)
            y_max = np.fmax(y_max, self.current_trajectory_point.longitude)

        plt.title(str(self.Status))
        plt.xlim(x_min-0.001, x_max+0.001)
        plt.ylim(y_min-0.001,y_max+0.001)
        plt.tight_layout()
        plt.show()
    def get_ground_truth(self):
        gt=[]
        for t in self.full_traj:
            gt.append(t.label)
        return gt
    def get_ground_mode(self):
        gt=[]
        for t in self.full_traj:
            gt.append(t.transportation_mode)
        return gt

    def cluster_trajectory_point(self, tp):

        self.number_of_processed_points=self.number_of_processed_points+1
        self.full_traj.append(tp)
        tp.trajectory_id=self.number_of_processed_points

        if self.current_cluster is not None:
            print("Curr cluster:",str(self.current_cluster.cluster_id))
        if self.noise_cluster is not None:
            print("noise cluster:",str(self.noise_cluster.get_cardinality()))
        if self.clusters is not None:
            for c in self.clusters:
                print("cluster:",str(c.cluster_id), "[",str(c.get_cardinality()),"]")
        if self.Status == "clustering":

            if self.window_buffer.cluster(self.current_trajectory_point):

                self.current_cluster.assign_point(self.window_buffer.trajectory_points[0])

                self.current_trajectory_point = self.window_buffer.trajectory_points[0]

                self.window_buffer.forward_window(tp)


            else:

                c = self.current_cluster.get_cardinality()

                if c <= self.min_points:
                    if self.current_cluster.trajectory_points is not None:
                        for tp0 in self.current_cluster.trajectory_points:

                            self.noise_cluster.assign_point(tp0)

                        self.current_cluster.reset()


                    self.current_cluster.assign_point(self.window_buffer.trajectory_points[0])


                    self.current_trajectory_point = self.window_buffer.trajectory_points[0]

                    self.window_buffer.forward_window(tp)


                else:


                    self.create_new_cluster()
                    self.current_cluster.assign_point(self.window_buffer.trajectory_points[0])
                    self.current_trajectory_point = self.window_buffer.trajectory_points[0]

                    self.window_buffer.forward_window(tp)



                pass
        else:
            if self.Status == "idle":
                print("idle read point:", str(tp.latitude), str(tp.longitude))
                # start of algorithm, create the first cluster and go to wait mode
                self.create_new_cluster()
                self.wait_for_window(tp)
            else:
                if self.Status == "wait":
                    print("wait read point:", str(tp.latitude), str(tp.longitude))
                    # wait for window buffer to start
                    self.wait_for_window(tp)
                else:
                    # raise error
                    raise Exception(
                        'ADB works in only three modes [idle, wait,cluster]. {} is an invalid state.'.format(
                            self.Status))

    def create_new_cluster(self,cid=None):
        # plt.show()
        # plt.figure()
        # plt.title("cluster", str(self.cluster_id))
        #if self.current_cluster is not None:
          #  self.current_cluster.plot()
          #  plt.show()
        cluster_id=cid
        if cid is None:
            cluster_id = self.cluster_id + 1
        print("Start cluster ", str(cluster_id))
        c = Cluster(cluster_id)
        self.cluster_id = cluster_id
        if self.clusters is None:

            self.clusters = np.array([c])
            self.current_cluster = c
        else:

            self.clusters = np.append(self.clusters, np.array([c]))
            self.current_cluster = c

        return c

    def __init__(self, window_size,sensitivity,min_points,speed_condition=True,bearing_cond=False):
        self.speed_condition=speed_condition
        self.bearing_cond=bearing_cond
        self.min_points=min_points
        self.sensitivity=sensitivity
        self.window_size = window_size
        self.noise_cluster = Cluster(-1)
        # print("new cluster")
