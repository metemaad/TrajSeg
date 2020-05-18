import numpy as np
import pandas as pd

from Trajlib2.SegmentationAlgorithms.SWS.SegmentBySWS import SegmentationBySWS
from Trajlib2.SegmentationAlgorithms.SegmentByGRASPUTS import SegmentationByGRASPUTS
from Trajlib2.SegmentationAlgorithms.SegmentByLabel import segment_by_label
from Trajlib2.SegmentationAlgorithms.SegmentByTime import SegmentationByTime
from Trajlib2.SegmentationAlgorithms.SegmentByWKMeans import SegmentationByWKMeans
from Trajlib2.SegmentationAlgorithms.SegmentByWSII import SegmentationByWSII
from Trajlib2.SegmentationAlgorithms.SegmentationByCBSMoT.SegmentByCBSMoT import SegmentationByCBSMoT
from Trajlib2.SegmentationAlgorithms.StayPointDetection import SegmentationByStayPointDetection


class TrajectorySegmentation:

    def __init__(self, row_trajectory=pd.DataFrame()):
        self.raw_trajectory = row_trajectory

    def return_row_data(self):
        return self.raw_trajectory

    def load_trajectory(self, trajectory):
        self.raw_trajectory = trajectory

    def load_data(self, **kwargs):
        print("This function is depricated and will be removed")
        # lat='lat',lon='lon',alt='alt',time_date='time_date',labels=['label1'],src='~/gps_fe/bigdata2_8696/ex_traj/5428_walk_790.csv',seperator=','
        lat = kwargs.get('lat', "lat")
        lon = kwargs.get('lon', "lon")
        alt = kwargs.get('alt', None)
        time_date = kwargs.get('time_date', "time_date")

        labels = kwargs.get('labels', "['target']")
        src = kwargs.get('src', "~/gps_fe/bigdata2_8696/ex_traj/5428_walk_790.csv")
        seperator = kwargs.get('seperator', ",")

        self.labels = labels
        # input data needs lat,lon,alt,time_date, [Labels]
        # ,nrows=80000
        # self.raw_data = self.raw_data.drop_duplicates(time_date)
        if type(src) is str:
            self.raw_trajectory = pd.read_csv(src, sep=seperator, parse_dates=[time_date], index_col=time_date)
        if type(src) is pd.DataFrame:
            self.raw_trajectory = src.copy()
            f = kwargs.get('time_format', "%Y-%m-%d %H:%M:%S")
            self.raw_trajectory[time_date] = pd.to_datetime(self.raw_trajectory[time_date], format=f)
            self.raw_trajectory.set_index(pd.DatetimeIndex(self.raw_trajectory[time_date]), inplace=True)
        # self.raw_data = self.raw_data.drop_duplicates(['t_user_id',time_date])
        # self.raw_data.set_index(time_date)

        self.raw_trajectory.rename(columns={(lat): ('lat')}, inplace=True)
        self.raw_trajectory.rename(columns={(lon): ('lon')}, inplace=True)
        if alt is not None:
            self.raw_trajectory.rename(columns={(alt): ('alt')}, inplace=True)
            self.hasAlt = True
        self.raw_trajectory.rename(columns={(time_date): ('time_date')}, inplace=True)
        # self.raw_data = self.raw_data.drop_duplicates(['t_user_id','time_date'])
        # self.raw_data = self.raw_data.set_index('time_date')

        # sort data first
        # self.raw_data=self.raw_data.sort_index()
        self.raw_trajectory['day'] = self.raw_trajectory.index.date

        # preprocessing
        # removing NaN in lat and lon

        self.raw_trajectory = self.raw_trajectory.loc[pd.notnull(self.raw_trajectory.lat), :]

        self.raw_trajectory = self.raw_trajectory.loc[pd.notnull(self.raw_trajectory.lon), :]

        for label in labels:
            self.raw_trajectory = self.raw_trajectory.loc[pd.notnull(self.raw_trajectory[label]), :]

        return self.raw_trajectory

    def multi_label_segmentation(self, labels=None, max_points=1000, max_length=False):

        if labels is None:
            labels = ['t_user_id', 'transportation_mode']
        segments = []
        start = 0
        end = self.raw_trajectory.shape[0] - 1

        print(self.raw_trajectory.shape)

        segments.append([start, end])

        for label in labels:
            new_segments = []
            for seg in range(len(segments)):
                start = segments[seg][0]
                end = segments[seg][1]

                stseg = self.raw_trajectory.iloc[start:end, :]
                s, sta = self.onelabelsegmentation(stseg, label)
                j = 0
                sn = []
                for x in range(len(s)):
                    # discritize if more than max_points=1000
                    # print(s[x][0] + start, s[x][1]+start,start)
                    leng = s[x][1] - s[x][0]
                    start2 = start + s[x][0]
                    if (max_length == False) & (leng >= max_points):
                        leng = s[x][1] - s[x][0]

                        idx = np.array(range(leng))
                        f = idx[idx % max_points == 0]
                        if len(f[1:]) == 0:
                            sn.append([s[x][0] + start, s[x][1] + start])
                            # print("good segment",sn)
                        else:
                            # print("idx",idx,f[1:],len(f[1:]))
                            if leng - f[1:][-1] < 10:
                                f[1:][-1] = leng

                            d = list(zip(f[:-1], f[1:]))
                            if f[1:][-1] != leng:
                                d.append((f[1:][-1], leng))
                            subtrajectories = {}
                            for i in d:
                                # print(i)
                                # s[i] =
                                sn.append([i[0] + start2, i[1] + start2])
                                #   print(j,i[0] + start2, i[1] + start2)
                                j = j + 1
                                # subtrajectories[i] = self.raw_data.iloc[i[0]:i[1], :]
                            del f
                            del idx
                    else:

                        # sn[j] = (s[x][0] + start, s[x][1] + start)
                        sn.append([s[x][0] + start, s[x][1] + start])
                        j = j + 1
                    # print(sn)
                # d=map(lambda (x, y):(x + start, y + start), s)
                #  print "start:",start,"s",s,"d:",d
                new_segments.extend(sn)
            #    st=nst
            # print "ddd",end, endd
            # new_segments.append([end,endd])
            # print(new_segments)
            # print end,endd

            segments = new_segments
        # s=[end,endd]

        # print segments
        trajectorySegments = {}
        # print len(segments)-1
        for seg in range(len(segments)):
            start = segments[seg][0]
            end = segments[seg][1]
            trajectorySegments[seg] = self.raw_trajectory.iloc[start:end, :]
        # seg=5682
        # print nst[0],len(nst)
        # print seg,nst[seg]
        return segments, trajectorySegments

    def segment_by_polygon(self):
        segments, trajectorySegments = None, None
        return segments, trajectorySegments

    def segment_by_label(self, label='transportation_mode'):

        segments, trajectory_segments = segment_by_label(self.raw_trajectory, label)

        seg_id = np.zeros(segments[len(segments) - 1][1])
        l = 0
        for s in segments:
            seg_id[s[0]:s[1] + 1] = l
            l += 1

        return seg_id

    def segment_by_wsii(self, tuning_set=None, tuning_set_ground_truth=None, wsii_params=None,
                        # window_size_params=[7], majority_vote_degree_params=[0.9], binary_classifier_params=[None],
                        window_size=7, majority_vote_degree=0.9, binary_classifier=None,kernel='Random Walk', verbose=False):

        # if window_size_params is None:
        #    window_size_params = [7]
        # if majority_vote_degree_params is None:
        #    majority_vote_degree_params = [0.9]
        ss = SegmentationByWSII(window_size=window_size, majority_vote_degree=majority_vote_degree,
                                    binary_classifier=binary_classifier,kernel=kernel)
        if tuning_set is not None:
            ss = ss.tuning(tuning_set, tuning_set_ground_truth,
                           wsii_params=wsii_params, verbose=verbose)
            # window_size_params=window_size_params,
            # majority_vote_degree_params=majority_vote_degree_params,
            # binary_classifier_params=binary_classifier_params)
            seg_id = ss.segment_id


        else:
            seg_id = ss.predict(self.raw_trajectory)

        best_param_dic = {'window_size': ss.window_size, 'majority_vote_degree': ss.majority_vote_degree,
                          'binary_classifier': ss.binary_classifier,'kernel':ss.kernel}
        return seg_id, best_param_dic

    def segment_by_RGRASP(self, tuning_set=None, tuning_set_ground_truth=None,
                          window_size_params=[7], majority_vote_degree_params=[0.9], binary_classifier_params=[None],
                          window_size=7, majority_vote_degree=0.9, binary_classifier=None):

        if window_size_params is None:
            window_size_params = [7]
        if majority_vote_degree_params is None:
            majority_vote_degree_params = [0.9]

        if tuning_set is not None:
            ss = SegmentationByWSII().tuning(tuning_set, tuning_set_ground_truth,
                                             window_size_params=window_size_params,
                                             majority_vote_degree_params=majority_vote_degree_params,
                                             binary_classifier_params=binary_classifier_params)
            seg_id = None
            best_param = (ss.window_size,
                          ss.majority_vote_degree,
                          ss.binary_classifier)
        else:
            ss = SegmentationByWSII(window_size=window_size, majority_vote_degree=majority_vote_degree,
                                    binary_classifier=binary_classifier)
            seg_id = ss.predict(self.raw_trajectory)
            best_param = (ss.window_size,
                          ss.majority_vote_degree,
                          ss.binary_classifier)
        return seg_id, best_param

    def segment_by_grasputs(self, tuning_set=None, tuning_set_ground_truth=None, grasputs_params=None,
                            feature_names=None,
                            alpha=0.3, partitioning_factor=0, max_iterations=30, min_time=6, jcs=0.3,
                            lat_col=None, lon_col=None, verbose=False):

        if tuning_set is not None:
            ss = SegmentationByGRASPUTS().tuning(tuning_trajectory=tuning_set,
                                                 tuning_ground_truth=tuning_set_ground_truth,
                                                 grasputs_params=grasputs_params, verbose=verbose)
            seg_id = ss.segment_id

        else:
            ss = SegmentationByGRASPUTS(lat_col=lat_col, lon_col=lon_col, feature_names=feature_names, alpha=alpha,
                                        partitioning_factor=partitioning_factor, max_iterations=max_iterations,
                                        min_time=min_time, join_coefficient=jcs, verbose=verbose)
            seg_id = ss.predict(self.raw_trajectory)

        best_param_dic = {'alpha': ss.alpha,
                          'partitioning_factor': ss.partitioning_factor,
                          'max_iterations': ss.max_iterations,
                          'min_time': ss.min_time,
                          'jcs': ss.join_coefficient}
        return seg_id, best_param_dic

    def segment_by_wkmeans(self, tuning_set=None, tuning_set_ground_truth=None,
                           wkmeans_params=None,
                           # num_k_param=None, delta_param=None,
                           num_k=None, delta=0, columns=['lat', 'lon'], verbose=False):

        # if delta_param is None:
        #    delta_param = [0]

        # if num_k_param is None and tuning_set is not None:
        #    mx = tuning_set.shape[0]

        #    num_k_param = list(range(1, mx))
        if tuning_set is not None:
            ss = SegmentationByWKMeans(columns=columns).tuning(tuning_set, tuning_set_ground_truth,
                                                               wkmeans_params=wkmeans_params,
                                                               verbose=verbose)
            # num_k_param=num_k_param, delta_param=delta_param)
            seg_id = ss.segment_id

        else:
            ss = SegmentationByWKMeans(num_k=num_k, delta=delta, columns=columns)
            seg_id = ss.predict(self.raw_trajectory)

        best_param_dic = {'num_k': ss.num_k,
                          'delta': ss.delta}
        return seg_id, best_param_dic

    def segment_by_spd(self, tuning_set=None, tuning_set_ground_truth=None,
                       spd_params=None,
                       theta_distance=1000, theta_time=500,
                       theta_distance_param=None, theta_time_param=None, verbose=False):

        # if theta_distance_param is None:
        #    theta_distance_param = [100, 200, 500, 1000, 2000]
        # if theta_time_param is None:
        #    theta_time_param = [60, 300, 600, 1200]
        if tuning_set is not None:
            ss = SegmentationByStayPointDetection().tuning(tuning_set, tuning_set_ground_truth, spd_params=spd_params,
                                                           verbose=verbose)
            seg_id = ss.segment_id
        else:
            ss = SegmentationByStayPointDetection(theta_distance=theta_distance, theta_time=theta_time)
            seg_id = ss.predict(self.raw_trajectory)

        best_param_dic = {'theta_time': ss.theta_time,
                          'theta_distance': ss.theta_distance}
        return seg_id, best_param_dic

    def segment_by_time(self, tuning_set=None, tuning_set_ground_truth=None, mean_time=250,
                        minimum_number_of_items_in_each_traj=10,
                        search_param=None):
        if search_param is None:
            search_param = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 180, 250, 350, 500]
        if tuning_set is not None:
            ss = SegmentationByTime().tuning(tuning_set=tuning_set, tuning_set_ground_truth=tuning_set_ground_truth,
                                             params=search_param)
            seg_id = ss.predict(self.raw_trajectory)
        else:
            ss = SegmentationByTime(mean_time=mean_time,
                                    minimum_number_of_items_in_each_traj=minimum_number_of_items_in_each_traj)
            seg_id = ss.predict(self.raw_trajectory)
        return seg_id

    def segment_cbsmot(self, tuning_set=None, tuning_set_ground_truth=None,
                       cbsmot_params=None,
                       max_dist=100, area=0.5, min_time=60, time_tolerance=60, merge_tolerance=100, verbose=False):

        if tuning_set is not None:

            ss = SegmentationByCBSMoT().tuning(tuning_set, tuning_set_ground_truth,
                                               cbsmot_params=cbsmot_params, verbose=verbose)
            seg_id = ss.segment_id

        else:
            ss = SegmentationByCBSMoT(max_dist=max_dist, area=area, min_time=min_time, time_tolerance=time_tolerance,
                                      merge_tolerance=merge_tolerance, verbose=verbose)
            seg_id = ss.predict(self.raw_trajectory)

        best_param_dic = {'max_dist': ss.max_dist, 'area': ss.area, 'min_time': ss.min_time,
                          'time_tolerance': ss.time_tolerance, 'merge_tolerance': ss.merge_tolerance}
        return seg_id, best_param_dic

    def segment_by_sws(self, tuning_set=None, tuning_set_ground_truth=None,
                       sws_params=None, epsilon=None, window_size=7, interpolation_kernel='linear', verbose=False,percentile=None):

        if tuning_set is not None:

            ss = SegmentationBySWS().tuning(tuning_trajectory=tuning_set, tuning_ground_truth=tuning_set_ground_truth,
                                            sws_params=sws_params, verbose=verbose)
            seg_id = ss.segment_id

        else:
            ss = SegmentationBySWS(epsilon=epsilon, window_size=window_size, interpolation_kernel=interpolation_kernel,
                                   verbose=verbose,percentile=percentile)
            seg_id = ss.predict(self.raw_trajectory, verbose=verbose)

        best_param_dic = {'interpolation_kernel': ss.interpolation_kernel, 'window_size': ss.window_size,
                          'epsilon': ss.epsilon,'percentile':ss.percentile}

        return seg_id, best_param_dic

    def __del__(self):
        del self.raw_trajectory
