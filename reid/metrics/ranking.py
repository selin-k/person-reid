import numpy as np
import os.path as osp
import shutil
import cv2
import os
import errno

def delete_if_exists(dirname):
    if os.path.exists(dirname) and os.path.isdir(dirname):
        shutil.rmtree(dirname)


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.

    Evaluation protocol as referenced in:
        - https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html

    Imported from: 
        - "https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py"

    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )


    indices = np.argsort(distmat, axis=1)
    # sorted_dist2 = distmat[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
  
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP




def open_world_eval(
    distmat, 
    q_pids, 
    g_pids, 
    thresholds=[0,1], 
    probes=None, 
    gallery=None, 
    visualize=False,
    width=128,
    height=256,
    save_dir_path='res_imgs'
):
    """
    Open world evaluation. 
        For each probe, check the best matching gallery instance surpasses a recognition threshold
        to identify strangers from known individuals.
        We want to evaluate the model's ability to identify known people as well as a tell them apart 
        from definite strangers.
    """
    delete_if_exists(save_dir_path)
    mkdir_if_missing(save_dir_path)
    b_acc_l = []
    for threshold in thresholds:

        print("Threshold: {}".format(threshold))
        save_dir=save_dir_path+"/threshold_"+str(threshold)
        delete_if_exists(save_dir)

        fn = tn = tp = fp = 0
        max_similarity_dist = 0
        min_similarity_dist = 1
        tpr_l = []
        fpr_l = []
        cur = ""

        for q_idx, probe_dists in enumerate(distmat):
            # each probe is a list of similarity values to everyone in the gallery.
            indices = np.argsort(probe_dists)
            best_match_idx = indices[0]
            best_match_similarity_dist = probe_dists[best_match_idx]
            max_similarity_dist = max(best_match_similarity_dist, max_similarity_dist)
            min_similarity_dist = min(best_match_similarity_dist, min_similarity_dist)

            q_pid = q_pids[q_idx]
            g_pid = g_pids[best_match_idx]
            id_match = q_pid == g_pid

            if best_match_similarity_dist > threshold:
                # we assume the probe is a stranger.
                if id_match:
                    # false negative detected
                    fn += 1
                    cur = "FN"
                else:
                    # true negative detected
                    tn += 1
                    cur = "TN"
            else:
                # best match is the identity of the probe.
                if id_match:
                    # true positive detected.
                    tp += 1
                    cur = "TP"
                else:
                    # false positive detected.
                    fp += 1
                    cur = "FP"

            if visualize:
                visualize_matching(
                    probes,
                    gallery,
                    q_idx,
                    best_match_idx,
                    width,
                    height,
                    save_dir,
                    cur,
                    q_pid,
                    g_pid,
                    best_match_similarity_dist
                )

            if q_idx % 20 == 0:
                p = tp+fn
                n = tn+fp
                tpr = 0 if p == 0 else tp/p
                tnr = 0 if n == 0 else tn/n
                fpr = 0 if n == 0 else fp/n
                tpr_l.append(tpr)
                fpr_l.append(fpr)


        p = tp+fn
        n = tn+fp
        tpr = 0 if p == 0 else tp/p
        fpr = 0 if n == 0 else fp/n
        tnr = 0 if n == 0 else tn/n
        b_acc = (tpr + tnr) / 2
        b_acc_l.append(b_acc)
                
                    
        print("Evaluation results:\n")
        print("TP:{}  FP:{}\nTN:{}  FN:{}\n".format(tp, fp, tn, fn))
        print("TPR:{} FPR:{}".format(tpr, fpr))
        print("Max similarity score: {} Min similarity score: {}\n".format(max_similarity_dist, min_similarity_dist))

                
    print("Thresholds: {}\nBalanced Accuracy:{}\n".format(thresholds, b_acc_l))
    

def visualize_matching(
    probes, 
    gallery, 
    q_idx, 
    best_match_idx, 
    width, 
    height, 
    save_dir, 
    category, 
    q_pid, 
    g_pid,
    best_match_similarity_dist
):
    GRID_SPACING = 10
    QUERY_EXTRA_SPACING = 90

    qimg_path = probes[q_idx][0]
    qimg_path_name = qimg_path[0] if isinstance(
        qimg_path, (tuple, list)
    ) else qimg_path
    qimg = cv2.imread(qimg_path)
    qimg = cv2.resize(qimg, (width, height))

    gimg_path = gallery[best_match_idx][0]
    gimg = cv2.imread(gimg_path)
    gimg = cv2.resize(gimg, (width, height))

    grid_img = 255 * np.ones(
        (
            height,
            2 * width + GRID_SPACING + QUERY_EXTRA_SPACING, 3
        ),
        dtype=np.uint8
    )
    grid_img[:, :width, :] = qimg
    start = width + GRID_SPACING + QUERY_EXTRA_SPACING
    end = 2 * width + GRID_SPACING + QUERY_EXTRA_SPACING
    grid_img[:, start:end, :] = gimg

    imname = osp.basename(osp.splitext(qimg_path_name)[0])
    cur_dir = save_dir + "/{}".format(category)
    mkdir_if_missing(cur_dir)
    imname += "_probe:{}_gallery:{}_dist{}".format(q_pid, g_pid,str(round(best_match_similarity_dist,2)))
    cv2.imwrite(osp.join(cur_dir, imname + '.jpg'), grid_img)

