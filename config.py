# -- coding: utf-8 --
import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=20,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=str,default='cuda:0', help='use cuda')
parser.add_argument('--seed', type=int, default=1235, help='random seed') #1235

# data in/out and dataset
parser.add_argument('--train_dataset_path',default = 'train.txt',help='fixed trainset root path')
parser.add_argument('--val_dataset_path',default = 'val.txt',help='fixed trainset root path')
parser.add_argument('--save_path',default='',help='save path of trained model')
parser.add_argument('--log_path',default='',help='save path of log')

# train
parser.add_argument('--batch_size', type=int, default=64,help='batch size of trainset')  
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',help='learning rate (default: 0.0001)') #sim 0.0001,其他都是0.002
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--decay', type=float, default=0.0, help='weight decay of AdamW') 

#DropKey
parser.add_argument('--use_DropKey', type=bool, default=False, help='use DropKey')
parser.add_argument('--TRRN_dropKey', type=float, default=0.4, help='ratio of dropkey')

#K-Fold
parser.add_argument('--K', type=int, default=10, help='value of Key')

#Model
parser.add_argument('--image_size', type=int, default=32, help='image size of T2T module')
parser.add_argument('--dim', type=int, default=128, help='dimension of the model')
parser.add_argument('--channels', type=int, default=1, help='computing dimensions of T2T')
parser.add_argument('--time_emb', type=str, default="One", help='Time embbeding methods')
parser.add_argument('--TRRN_depth', type=int, default=7, help='T2T depths')
parser.add_argument('--mode', type=str, default="pre_post", help='mode of fusion')
parser.add_argument('--num_classes', type=int, default=5, help='mode of fusion')
parser.add_argument('--FCL_gamma', type=int, default=2, help='parameter gamma for Focal loss')
parser.add_argument('--TCL_m', type=float, help='parameter margin for contrastive loss')

#Loss function
parser.add_argument('--loss_mode', type=str, default="fixed", help='fixed or adapted')

#data type
parser.add_argument('--data_type', type=str, default="OAI", help='ESCC or OAI')
parser.add_argument('--loss_type', type=str, default="ce", help='ce or focal')

args = parser.parse_args()