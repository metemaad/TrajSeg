from old_code.GRASP_UTS import grasp_uts_execute
from old_code.time_calculation import Clock
from old_code.UnsupervisedEvaluation import calculate_purity, calculate_coverage
from cost.cost_function import calculate_cost
from cost.DistanceMetrics import get_feature_bounds
from old_code import settings
from sklearn.model_selection import ParameterGrid




def execute(trajectories, proper_segmentation, max_distances,min_time, join_coefficient, partitioning_factor,alpha, max_iterations):
   # max_distances = read_max_distances(DIRECTORY)
   # trajectories, proper_segmentation = read_trajectories(DIRECTORY)
    total_segs = 0
    count = 0
    c = Clock()
    purity = 0
    min_time = min_time * 3600
    cost_sum = 0
    purity_weighted_by_segment_sum = 0
    coverage_sum_total = 0
    #trajectories = trajectories[:25]

    for t in trajectories:
        # t = trajectories[0]
        count += 1
        print("Trajectory " + str(count) + " of " + str(len(trajectories)))
        md = float(max_distances[t.tid])
        #c.start("GRASP UTS")
        segments = grasp_uts_execute(t, md, min_time, join_coefficient, partitioning_factor, alpha, max_iterations)
        #c.stop()
        purity_sum = 0
        coverage_sum = 0
        for s in segments:
            purity_sum += calculate_purity(s)
            coverage_sum += calculate_coverage(s, proper_segmentation[t.tid])
            total_segs+=1
        coverage_sum_total += coverage_sum
        purity_weighted_by_segment_sum += purity_sum
        cost_sum += calculate_cost(segments, md, get_feature_bounds(t))
        purity += (purity_sum / len(segments)) * 100

    #print("Average time for a Trajectory: " + str(c.get_avg()))
    """ print("Total Time (Seconds): " + str(c.get_avg() * len(trajectories)))
    print("Minimum time for a Trajectory: %f " % c.min)
    print("Maximum time for a Trajectory: %f " % c.max)"""

    p = (purity_weighted_by_segment_sum / total_segs) * 100
    c = (coverage_sum_total / total_segs) * 100
    co = cost_sum / len(trajectories)
    ts = total_segs
    harmonic_mean = 2 * (p*c)/(p+c)
    print("Average Purity: %f " % ((purity_weighted_by_segment_sum / total_segs) * 100))
    print("Average Coverage: %f " % ((coverage_sum_total / total_segs) * 100))
    print("Harmonic Mean: %f " % harmonic_mean)
    print("Average Cost: %f" % (cost_sum / len(trajectories)))
    print("Total Segments: " + str(total_segs))
    print("*************************************************************************************")

    return p, c, harmonic_mean, co, ts


def train(trajectories,proper_segmentation, max_distances, file_name):
    # NOTE : Adjust the parameters however you like for testing
    # Another Note : The way this is set up currently, it is going to take a REALLY LONG time
    alpha = settings.alpha
    partitioning_factor = settings.partitioning_factor
    max_iterations = settings.max_iterations
    min_times = settings.min_times
    jcs = settings.join_coefficients
    grid = {'alpha':alpha, 'partitioning_factor': partitioning_factor, 'max_iterations':max_iterations,'min_time':min_times,'jcs':jcs}
    parm_grid = ParameterGrid(grid)

    row = "Min Time, Join Coefficient, alpha, partioning_factor, max_iterations, Purity, Coverage, Harmonic Mean, Cost, Number Of Segments\n"
    f = open(file_name, "w+")
    f.write(row)
    f.close()
    harmonic_mean_order = []

    for option in parm_grid:
        t = option['min_time']
        jc = option['jcs']
        a = option['alpha']
        pf = option['partitioning_factor']
        max_iter = option['max_iterations']

        print("Min Time: %d | Join Coefficient: %.2f | Alpha: %.2f | Partitioning Factor: %.2f | Max Iterations: %d" % (t,jc,a,pf,max_iter))

        p, c, harmonic_mean, co, ts = execute(trajectories,proper_segmentation,max_distances,t,jc,pf,a,max_iter)
        row = "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (t,jc,a,pf,max_iter,p,c,harmonic_mean,co,ts)
        f = open(file_name, "a")
        f.write(row)
        f.close()
        harmonic_mean_order.append((co,t, jc,a,pf,max_iter))

    harmonic_mean_order.sort(key=lambda x:x[0],reverse=False )
    return harmonic_mean_order[0]


def test(trajectories, proper_segmentation, max_distances, file_name, alpha, partitioning_factor, min_time, join_coefficient,max_iterations):
    row = "Min Time, Join Coefficient, alpha, partioning_factor, max_iterations, Purity, Coverage, Harmonic Mean, Cost, Number Of Segments, trajectories\n"
    f = open(file_name, "w+")
    f.write(row)
    f.close()
    p, c, harmonic_mean, co, ts = execute(trajectories, proper_segmentation, max_distances, min_time, join_coefficient, partitioning_factor, alpha, max_iterations)
    row = "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,\"%s\"\n" % (min_time, join_coefficient, alpha, partitioning_factor, max_iterations,p,c,harmonic_mean,co,ts,str(trajectories))
    f = open(file_name, "a")
    f.write(row)
    f.close()
    return (co, p, c, harmonic_mean, ts)



