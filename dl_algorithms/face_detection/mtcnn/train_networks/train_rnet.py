
import argparse
import sys
#from dface.core.imagedb import ImageDB
#import dface.train_net.train as train
#import dface.config as config
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_dataloader import image_database
from base_train import train_pnet, train_rnet
import yaml
f = open('config.yaml', encoding='utf-8')
configure = yaml.load(f)



def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01, batch_size=128, use_cuda=False):

    imagedb = image_database(annotation_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)

    train_rnet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr, use_cuda=use_cuda)

def parse_args():
    parser = argparse.ArgumentParser(description='Train  RNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--anno_file', dest='annotation_file',
                        default=os.path.join(configure["ANNO_STORE_DIR"],configure["RNET_TRAIN_IMGLIST_FILENAME"]), help='training data annotation file', type=str)
    parser.add_argument('--model_path', dest='model_store_path', help='training model store directory',
                        default=configure["MODEL_STORE_DIR"], type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=configure["END_EPOCH"], type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=configure["TRAIN_LR"], type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=configure["TRAIN_BATCH_SIZE"], type=int)
    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=configure["USE_CUDA"], type=bool)
    parser.add_argument('--prefix_path', dest='', help='training data annotation images prefix root path', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('train Rnet argument:')
    print(args)


    train_net(annotation_file=args.annotation_file, model_store_path=args.model_store_path,
                end_epoch=args.end_epoch, frequent=args.frequent, lr=args.lr, batch_size=args.batch_size, use_cuda=args.use_cuda)
