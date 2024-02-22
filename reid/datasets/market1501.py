from .dataset import ReidDataset, check_before_run

import os.path as osp
import random
import re
import glob


class Market1501(ReidDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
        - cameras: 6.
    """


    def __init__(self, root='', open_set=False, open_set_gallery_ratio=1, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        check_before_run(required_files)

        self.train = self.__process_dir(self.train_dir, relabel=True)
        self.query = self.__process_dir(self.query_dir, relabel=False)
        self.gallery = self.__process_dir(self.gallery_dir, relabel=False)

        print("=> Market1501 loaded")
        self._print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self._get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self._get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self._get_imagedata_info(
            self.gallery)


        self.num_openset_pids = None
        self.open_set_gallery = None
        self.open_set_probes = None
        self.open_set_pids = None
        self.all_pids = None

        if open_set:
            # let the open set be the given ratio of the identities in the gallery
            self.num_openset_pids = int(self.num_gallery_pids * open_set_gallery_ratio)

            # select that many random pids from the gallery and let the open set be these only
            pids = set()
            for _ ,pid, _ in self.gallery:
                pids.add(pid)

            random.seed(10)
            rnd_pids = random.sample(pids, self.num_openset_pids)
            open_set_gallery = []
            for img_path, pid, camid in self.gallery:
                if pid in rnd_pids:
                    open_set_gallery.append((img_path,pid,camid))
    

            # select the probes with the same pids as the gallery individuals.
            open_set_probes = []
            for img_path, pid, camid in self.query:
                if pid in rnd_pids:
                    open_set_probes.append((img_path,pid,camid))


            self.open_set_gallery = open_set_gallery
            self.open_set_probes = open_set_probes
            self.open_set_pids = rnd_pids
            self.all_pids = pids

            print("open pids len:{}".format(len(rnd_pids)))


    def __process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset



