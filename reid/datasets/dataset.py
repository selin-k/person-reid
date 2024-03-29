from PIL import Image
import os.path as osp
from torch.utils.data import Dataset




def check_before_run(required_files):
        """Checks if required files exist before going deeper."""
        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))



class ReidDataset(object):

    def _get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams


    def _print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self._get_imagedata_info(
            train)
        num_query_pids, num_query_imgs, num_query_cams = self._get_imagedata_info(
            query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self._get_imagedata_info(
            gallery)

        print("  Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(
            num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(
            num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(
            num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


    
class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __read_image(self, img_path):
        """Keep reading image until success.
        This can avoid IOError incurred by heavy IO processes."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'.".format(img_path))
                pass
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = self.__read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path