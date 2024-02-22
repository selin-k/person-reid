import numpy as np
import os.path as osp
import cv2

from reid.engine.utils import mkdir_if_missing

__all__ = ['visualize_ranked_results']

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(
    distmat, dataset, width=128, height=256, save_dir='', topk=10, top_correct_ranks=None
):
    """Visualizes ranked results.

    Imported from: "https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/accuracy.py"

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid, dsetid).
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    if top_correct_ranks is None:
        # Not skipping anything
        top_correct_ranks = 0
        
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)


    # For all query images, loop the gallery images that occur as the top indices of the query
    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx][:3]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

    
        qimg = cv2.imread(qimg_path)
        qimg = cv2.resize(qimg, (width, height))
        qimg = cv2.copyMakeBorder(
            qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        # resize twice to ensure that the border width is consistent across images
        qimg = cv2.resize(qimg, (width, height))
        num_cols = topk + 1
        grid_img = 255 * np.ones(
            (
                height,
                num_cols*width + topk*GRID_SPACING + QUERY_EXTRA_SPACING, 3
            ),
            dtype=np.uint8
        )
        grid_img[:, :width, :] = qimg
    
        rank_idx = 1
        skip_image = False
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx][:3]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid

                # If the top_correct_ranks match is correct, skip visualization
                if rank_idx <= top_correct_ranks and matched:
                    skip_image = True
                    break

        
                border_color = GREEN if matched else RED
                gimg = cv2.imread(gimg_path)
                gimg = cv2.resize(gimg, (width, height))
                gimg = cv2.copyMakeBorder(
                    gimg,
                    BW,
                    BW,
                    BW,
                    BW,
                    cv2.BORDER_CONSTANT,
                    value=border_color
                )
                gimg = cv2.resize(gimg, (width, height))
                start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                end = (
                    rank_idx+1
                ) * width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                grid_img[:, start:end, :] = gimg
            

                rank_idx += 1
                if rank_idx > topk:
                    break

        if not skip_image:
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
