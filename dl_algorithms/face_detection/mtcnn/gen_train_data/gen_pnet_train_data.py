import argparse
import os
import numpy as np
import cv2
import numba as nb


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
        box_idx =0
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

        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
            # generate negative examples that have overlap with gt
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = util.IoU(crop_box, boxes)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s_neg.jpg" % neg_idx)
                    f2_neg.write(save_file + ' 0\n')# gen neg label
                    cv2.imwrite(save_file, resized_im)# save the random neg image
                    neg_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                   continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                # print(ny1 , ny2, nx1 , nx2)
                # cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if util.IoU(crop_box, box_) >= 0.70:
                    save_file = os.path.join(pos_save_dir, "%s_postive.jpg" % post_idx)
                    f1_post.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    post_idx += 1
                elif util.IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s_part.jpg" % det_idx)
                    f3_part.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    det_idx += 1

                box_idx += 1
                print("%s images finish generation, pos: %s part: %s neg: %s" % (idx, post_idx, det_idx, neg_idx))
    f1_post.close()
    f2_neg.close()
    f3_part.close()

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