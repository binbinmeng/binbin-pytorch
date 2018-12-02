import argparse
import os
import numpy as np
import cv2
import numba as nb
from utils import util


def gen_pnet_train_data(store_pnet_data_dir,anno_file,prefix):
    neg_save_dir = os.path.join(store_pnet_data_dir, "12/negative")
    pos_save_dir = os.path.join(store_pnet_data_dir, "12/positive")
    part_save_dir = os.path.join(store_pnet_data_dir, "12/part")
    # create the three dir
    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    save_dir = os.path.join(store_pnet_data_dir, "pnet")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    post_save_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_POSTIVE_ANNO_FILENAME)
    neg_save_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_NEGATIVE_ANNO_FILENAME)
    part_save_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_PART_ANNO_FILENAME)

    f1_post = open(post_save_file, 'w')
    f2_neg = open(neg_save_file, 'w')
    f3_part = open(part_save_file, 'w')

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    num = len(annotations)
    print("%d images in total" % num)

    idx = 0  # used for count total images number
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        image_path = os.path.join(prefix, annotation[0])
        bbox = list(map(float, annotation[1:]))  # get four point values of bbox
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)

        print(image_path)
        if image_path is None:
            continue

        img = cv2.imread(image_path)

        if img is None:
            continue

        post_idx = 0
        neg_idx = 0
        det_idx = 0

        idx += 1
        if idx % 100 == 0:
            print(idx, "images finish generation")

        height, width, channel = img.shape

        neg_num = 0  # static negative image num random generated
        while neg_num < 50:
            size = np.random.randint(12, min(width, height) / 2)  # random generate a size
            nx = np.random.randint(0, width - size)  # random generate a start x
            ny = np.random.randint(0, height - size)  # random generate a start y
            crop_box = np.array([nx, ny, nx + size, ny + size])  # crop a random box size

            Iou = util.IoU(crop_box, boxes)  # judge the iou if is ok
            if np.max(Iou) < 0.3:
                cropped_im = img[ny: ny + size, nx: nx + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)  # PNet input is 12

                save_file = os.path.join(neg_save_dir, "%s_neg.jpg" % neg_idx)
                f2_neg.write(save_file + ' 0\n')  # gen neg label
                cv2.imwrite(save_file, resized_im)  # save the random neg image
                neg_idx += 1
                neg_num += 1

def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--face_traindata_store', dest='traindata_store',
                        help='face train data temporary folder,include 12,24,48/postive,negative,part,landmark',
                        default='../data/wider/', type=str)
    parser.add_argument('--anno_file', dest='annotation_file', help='wider face original annotation file',
                        default=os.path.join(config.ANNO_STORE_DIR,"wider_origin_anno.txt"), type=str)
    parser.add_argument('--prefix_path', dest='prefix_path', help='annotation file image prefix root path',
                        default='', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gen_pnet_train_data(args.traindata_store,args.annotation_file,args.prefix_path)