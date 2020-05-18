from old_code.read_trajectories import read_max_distances
from old_code.prepare import randomize_array, prepare_chunks, prepare_training_testing
from old_code.execute import train,test
from old_code.time_calculation import  Clock
import os
from old_code import settings

DATASET = settings.dataset
DIRECTORY = settings.directory

def cross_validation():
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    samples, proper_seg, feat_bounds = settings.read_dataset()
    md = read_max_distances(DATASET)
    N = 10
    samples = randomize_array(samples)
    chunks = prepare_chunks(samples,N)
    row = "Min Time, Join Coefficient, alpha, partioning_factor, max_iterations, Purity, Coverage, Harmonic Mean, Cost, Number Of Segments, Testing Trajectories, train time, test time, total time, train file name, test file name\n"
    test_name_all = DIRECTORY + DATASET + "_all_test.csv"
    f = open(test_name_all, "w+")
    f.write(row)
    f.close()
    for i in range(0,N):
        print('Training Set ' + str(i + 1))
        train_clock = Clock()
        test_clock = Clock()
        train_name = DIRECTORY + DATASET + str(i) + "_train.csv"
        test_name = DIRECTORY + DATASET + str(i) + "_test.csv"
        large_sample, small_sample = prepare_training_testing(chunks, i)
        train_clock.start('train')
        best_parameters = train(large_sample, proper_seg, md, train_name)
        train_clock.stop()
        train_time = train_clock.last_time
        # (co,t, jc,a,pf,max_iter
        test_clock.start('test')
        results = test(small_sample, proper_seg, md, test_name, best_parameters[3], best_parameters[4],
                       best_parameters[1],
                       best_parameters[2], best_parameters[5])
        test_clock.stop()
        test_time = test_clock.last_time
        # (co, p, c, harmonic_mean, ts)
        row = "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,\"%s\",%f,%f,%f,%s,%s\n" % (
            best_parameters[1], best_parameters[2], best_parameters[3], best_parameters[4], best_parameters[5],
            results[1],
            results[2], results[3], results[0], results[4], str(small_sample), train_time, test_time,
            (train_time + test_time),
            train_name, test_name)
        f = open(test_name_all, "a")
        f.write(row)
        f.close()
if __name__ =='__main__':
    cross_validation()