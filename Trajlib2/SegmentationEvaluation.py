from numpy import add, array, mean, maximum, where, multiply
from operator import itemgetter
from sklearn.metrics import precision_score


def purity(ground_truth_label, generated_segment_id):
    # P(thera_T, landa_L)
    # ground_truth=label
    # theta_t =generated segment_id.
    # Landa_L = set of labels by subject matter expert. -> ground_truth _label
    # T = number of discovered segments-> len(set(generated_segment_id))
    # L = the number of expert labels ->len(set(ground_truth_label)
    # Nij= Number of trajectory points inside segment Theta_i with label Landa_i
    # Ni= total number of points found for the segment theta_i
    theta_t = generated_segment_id
    landa_l = ground_truth_label

    avg = []
    landa_l = array(landa_l)
    theta_t = array(theta_t)
    for ts in set(theta_t):  # i=1..T, ts=i
        ma = 0  # sum value
        g = landa_l[(where(theta_t == ts)[0])]
        for tp in set(g):
            _ = len(where(g == tp)[0])
            if _ > ma:
                ma = _
        if ts != -1:
            avg.append(ma * 1.0 / len(g))
    return avg, mean(array(avg)), len(avg)


def coverage(ground_truth_segment_id, generated_segment_id):
    # ground_truth=tsid
    # kisi_v = the set of segments by subject matter expert
    # N_kisitheta= number of traj points of the segment theta_j that intersected the segment kisi_i
    # Ni= total number of points the segment kisi_i

    kisi_v = ground_truth_segment_id
    theta_t = generated_segment_id
    cov = []
    theta_t = array(theta_t)
    kisi_v = array(kisi_v)
    l2 = []
    for ts in set(kisi_v):
        mx = 0
        g = theta_t[(where(kisi_v == ts)[0])]
        for l in set(g):
            _ = len(where(g == l)[0])
            if mx <= _:
                mx = _
                l2.append(l)

        cov.append(mx * 1.0 / len(g))

    return cov, mean(array(cov))


def harmonic_mean(segments, tsid, label):
    cov = coverage(ground_truth_segment_id=tsid,
                   generated_segment_id=segments)[1]
    pur = purity(ground_truth_label=label,
                 generated_segment_id=segments)[1]
    return (2 * cov * pur) / (cov + pur)


def report(segments, tsid, label, verbose=True):
    p = purity(generated_segment_id=segments,
               ground_truth_label=label)[1]

    c = coverage(generated_segment_id=segments,
                 ground_truth_segment_id=tsid)[1]

    h = harmonic_mean(segments, label=label, tsid=tsid)
    k_segment1 = len(set(segments))
    k_tsid = len(set(tsid))
    print("Number of segments:", k_tsid, '->', k_segment1)

    segments = convert_segment_id_to_original_tsid(ground_truth_tsid=tsid, generated_segment_id=segments)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import cohen_kappa_score
    acc = accuracy_score(segments, tsid)
    kappa = cohen_kappa_score(segments, tsid)
    # if verbose:
    print("Purity:", p)
    print("Coverage:", c)
    print("Harmonic Mean:", h)
    print("Accuracy:", acc)
    print("Kappa:", kappa)
    p2 = purity(generated_segment_id=segments,
                ground_truth_label=label)[1]

    c2 = coverage(generated_segment_id=segments,
                  ground_truth_segment_id=tsid)[1]

    h2 = harmonic_mean(segments, label=label, tsid=tsid)
    print("Purity*:", p2)
    print("Coverage*:", c2)
    print("Harmonic Mean*:", h2)
    pr = precision_score(tsid, segments, average='micro')
    print("precision_scores:", pr)
    from sklearn.metrics import recall_score
    re = recall_score(tsid, segments, average='micro')
    print("recalls:", re)
    from sklearn.metrics import jaccard_score
    j = jaccard_score(tsid, segments, average='micro')
    print("jaccard:", j)
    k_segment2 = len(set(segments))
    k_tsid = len(set(tsid))
    print("Number of segments:", k_tsid, '->', k_segment2)

    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score
    from sklearn.metrics import v_measure_score
    homo=homogeneity_score(tsid,segments)
    comp=completeness_score(tsid,segments)
    v_measure=v_measure_score(tsid,segments)
    print("Homogenity:",homo)
    print("Completeness:",comp)
    print("v_measure:",v_measure)

    return p, c, h, k_segment1, k_segment2, k_tsid, acc, kappa, pr, re, j, p2, c2, h2,homo,comp,v_measure


def harmonic_mean2(segments, tsid, label):
    # print(len(ground_truth),len(prediction))
    # print(np.unique(ground_truth))
    # print(np.unique(prediction))
    cov = coverage(segments, tsid)[0]
    pur = purity(segments, label)[0]
    ##print("cov",len(cov),"pur",len(pur),len(pur)-len(cov))
    h = multiply([2] * len(cov), multiply(cov, pur)) / add(cov, pur)
    return h, mean(h)


def adjusted_purity(ground_truth, labels):
    avg = []
    ground_truth = array(ground_truth)
    labels = array(labels)
    k = []
    for ts in set(labels):
        ma = 0
        g = ground_truth[(where(labels == ts)[0])]
        for tp in set(g):
            _ = len(where(g == tp)[0])
            if _ > ma:
                ma = _
                k.append(tp)
        if ts != -1:
            avg.append(ma * 1.0 / len(g))
    ouliers = [-1]
    mm = maximum(len(set(ground_truth) - set(ouliers)), len(set(labels) - set(ouliers)))
    # print(mm)
    for i in range(mm - len(avg)):
        avg.append(0.0)
    # print("k",len(set(k)),len(set(labels)),len(set(labels))-len(set(k)),len(avg))
    return avg, mean(array(avg)), len(avg)


def adjusted_coverage(ground_truth, labels):
    cov = []
    labels = array(labels)
    ground_truth = array(ground_truth)
    l2 = []
    for ts in set(ground_truth):
        mx = 0
        g = labels[(where(ground_truth == ts)[0])]
        for l in set(g):
            _ = len(where(g == l)[0])
            # print(ts,l)
            if mx <= _:
                mx = _
                l2.append(l)

        cov.append(mx * 1.0 / len(g))
    ouliers = [-1]
    mm = maximum(len(set(ground_truth) - set(ouliers)), len(set(labels) - set(ouliers)))
    # print(mm)
    for i in range(mm - len(cov)):
        cov.append(0.0)
    # print(len(set(set(labels) - set(l2))),set(set(labels) - set(l2)))
    return cov, mean(array(cov))


def adjusted_harmonic_mean(ground_truth, prediction):
    # print(len(ground_truth),len(prediction))
    # print(np.unique(ground_truth))
    # print(np.unique(prediction))
    cov = adjusted_coverage(ground_truth, prediction)[0]
    pur = adjusted_purity(ground_truth, prediction)[0]
    # print("cov",len(cov),"pur",len(pur),len(pur)-len(cov))
    h = multiply([2] * len(cov), multiply(cov, pur)) / add(cov, pur)
    return h, mean(h)


def convert_segment_id_to_original_tsid(ground_truth_tsid, generated_segment_id):
    ground_truth_tsid = array(ground_truth_tsid)
    generated_segment_id = array(generated_segment_id)

    tsid_set = list(set(ground_truth_tsid))
    res = {}
    for gs in tsid_set:

        idx = where(ground_truth_tsid == gs)[0]
        seg = generated_segment_id[idx]
        dic = {}
        for ts in list(set(seg)):
            idx = where(seg == ts)[0]
            dic[ts] = len(seg[idx])
        m = max(dic.items(), key=itemgetter(1))[0]
        res[(gs, m)] = dic[m]

    sorted_x = sorted(res.items(), key=itemgetter(1))[::-1]
    final_label = {}

    lx = len(tsid_set)
    while lx > 0 and len(sorted_x) > 0:

        p = sorted_x[0]
        final_label[p[0][1]] = p[0][0]

        new_dic = {}
        for d in sorted_x:
            if d[0][1] != p[0][1]:
                new_dic[d[0]] = d[1]
        sorted_x = sorted(new_dic.items(), key=itemgetter(1))[::-1]
        lx -= 1
    # print(tsid_set)
    # print(list(set(generated_segment_id)))
    # print(list(set(generated_segment_id) - set(final_label.keys())))
    # print(final_label)
    new_class = array(list(tsid_set)).max() + 1
    for i in range(len(generated_segment_id)):
        generated_segment_id[i] = final_label.get(generated_segment_id[i], -1 * new_class)
    generated_segment_id = array(generated_segment_id)

    return generated_segment_id
