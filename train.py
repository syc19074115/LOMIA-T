from functools import partial
import os
import random,gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
from dataset.ESCC import train_dataset, val_dataset
from torch.utils.data import DataLoader
from utilis.scheduler import CosineAnnealingLRWarmup, LRScheduler
from tqdm import tqdm
import config
from main_model.LOMIA_T import LOMIA_T

# os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8' 


def Ordinal_loss(outputs, labels, device):
    softmax_op = torch.nn.Softmax(1)
    prob_pred = softmax_op(outputs)

    def set_weights():
        init_weights = np.array([[1, 3, 5, 7, 9],
                                 [3, 1, 3, 5, 7],
                                 [5, 3, 1, 3, 5],
                                 [7, 5, 3, 1, 3],
                                 [9, 7, 5, 3, 1]], dtype=np.float32)
        """ init_weights = np.array([[1, 3, 5, 7, 9, 11, 13],
                                 [3, 1, 3, 5, 7, 9, 11],
                                 [5, 3, 1, 3, 5, 7, 9],
                                 [7, 5, 3, 1, 3, 5, 7],
                                 [9, 7, 5, 3, 1, 3, 5],
                                 [11, 9, 7, 5, 3, 1, 3],
                                 [13, 11, 9, 7, 5, 3, 1]], dtype=np.float32) """

        adjusted_weights = init_weights + 1.0
        np.fill_diagonal(adjusted_weights, 0)

        return adjusted_weights
    cls_weights = set_weights()

    batch_num, class_num = outputs.size()
    class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
    labels_np = labels.data.cpu().numpy()
    for ind in range(batch_num):
        class_hot[ind, :] = cls_weights[labels_np[ind], :]
    class_hot = torch.from_numpy(class_hot)
    class_hot = torch.autograd.Variable(class_hot).to(device)

    loss = torch.sum((prob_pred * class_hot)**2) / batch_num
    # loss = torch.mean(prob_pred * class_hot)

    return loss

def model_load(net,device,type):
    from collections import OrderedDict
    if type == "ESCC":
        checkpoint_pre = torch.load('/pretrain_TRRN_ESCC.pth',map_location=device)
    net.TRRN_pre.load_state_dict(checkpoint_pre)

    if type == "ESCC":
        checkpoint_post = torch.load('/pretrain_TRRN_ESCC.pth',map_location=device)
    net.TRRN_post.load_state_dict(checkpoint_post, strict=False)

    return net

def randseed(args):
    torch.cuda.empty_cache()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

def main():
    args = config.args
    randseed(args)
    save_path = args.save_path
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    ckpts = args.save_path
    os.makedirs(ckpts, exist_ok=True)

    #tensorboard
    tb_writer = SummaryWriter(args.log_path)  # 创立tensorboard，日志为空则其自动创立
    # args.seed += 51
    randseed(args)
    
    #model
    net = LOMIA_T(image_size=args.image_size, dim=args.dim, channels=args.channels, time_emb=args.time_emb, TRRN_depth=args.TRRN_depth, TRRN_dropKey=args.TRRN_dropKey, use_DropKey=args.use_DropKey,mode=args.mode ,num_classes=args.num_classes, data_type=args.data_type)
    # net = model_load(net,device,args.data_type)

    #set to device
    net = net.to(device)

    #loss function
    from utilis.loss import ContrastiveLoss_euc,FocalLoss
    # FCL_alpha = torch.tensor([0.5, 0.5])
    if args.loss_type == "focal":
        FCL = FocalLoss(class_num=args.num_classes, gamma=args.FCL_gamma).to(device)
    elif args.loss_type == "ce":
        FCL = nn.CrossEntropyLoss()
    TCL = ContrastiveLoss_euc(margin=args.TCL_m).to(device)

    #daraset and  dataloader
    if args.data_type == "ESCC":
        train_ds = train_dataset.Train_Dataset(args)
        val_ds = val_dataset.Val_Dataset(args)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size, num_workers=args.n_threads)

    #optimizer
    #optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5.0e-4)
    optimizer = torch.optim.AdamW(net.parameters(),lr=args.lr,betas=(0.9, 0.999),weight_decay=args.decay) #0.03
    # optimizer = torch.optim.RAdam(net.parameters(),lr=args.lr,betas=(0.9, 0.999),weight_decay=args.decay)
    # optimizer = torch.optim.NAdam(net.parameters(),lr=args.lr,betas=(0.9, 0.999),weight_decay=args.decay)
    lr_scheduler_warmup = CosineAnnealingLRWarmup(optimizer,
                                                 T_max=args.epochs,
                                                 eta_min=1.0e-8, #1.0e-6时训练vgg达到最优
                                                 last_epoch=-1,
                                                 warmup_steps=args.warmup,
                                                 warmup_start_lr=1.0e-8)
    #lr_scheduler = LRScheduler(args.lr, 5)
    
    if args.loss_mode == "fixed":
        Weightloss1 = 0.01  #ESCC 0.01
        Weightloss2 = 1-Weightloss1
    elif args.loss_mode == "adapted":
        #Grad adapted adjusted loss function parameters
        Weightloss1 = torch.tensor(torch.FloatTensor([]), requires_grad=True) #记得给一个初始化的值
        Weightloss2 = torch.tensor(torch.FloatTensor([]), requires_grad=True)
        weight_params = [Weightloss1, Weightloss2]
        weight_opt = torch.optim.Adam(weight_params, lr=args.lr)
        weight_opt.zero_grad()

    optimizer.zero_grad()  #
    
    best_train_loss = 100000
    best_val_loss = 100000
    best_val_acc = 0
    for epoch in range(args.epochs):
        net.train()  # 设置成训练模式，作用其实不大，对Dropout、Batchnorm层起作用
        print("=======Epoch:{}=======lr:{}".format(epoch, lr_scheduler_warmup.get_last_lr()))
        # print("=======Epoch:{}=======".format(epoch))
        total_train_loss = 0
        total_train_acc = 0
        # for i, (pre, post, label_pre, label_post, TCL_label) in tqdm(enumerate(train_dl), total=len(train_dl)):
        for i, (pre, post, label_pre, label_post, TCL_label) in enumerate(train_dl):
            pre = pre.to(device)
            post = post.to(device)
            labels = label_post.long().to(device)
            TCL_label = TCL_label.to(device)       
            optimizer.zero_grad()
            logits, t_pre, t_post, t_crosspre, t_crosspost = net(pre, post)
            if isinstance(logits, tuple):
                logits = logits[0]
            _, preds = torch.max(logits.data, 1)
            accuracy = torch.sum(preds == labels.data)
            total_train_acc += accuracy
            TCL_loss = TCL(t_pre, t_post, TCL_label)
            FCL_loss = FCL(logits, labels.view(-1))
            # FCL_loss = Ordinal_loss(logits, labels, device) 
            # print(TCL_loss, FCL_loss)
            if args.loss_mode == "fixed":
                train_loss = Weightloss1 * TCL_loss + Weightloss2 * FCL_loss
                # print(FCL_loss.shape)
            elif args.loss_mode == "adapted":
                train_loss = weight_params[0] * TCL_loss + weight_params[1] * FCL_loss
                weight_opt.zero_grad()
                weight_opt.step() 
            total_train_loss += train_loss
            train_loss.backward()
            optimizer.step()
        lr_scheduler_warmup.step()

        print("model : Train loss = {}, Train acc = {}".format(total_train_loss/len(train_dl), total_train_acc / len(train_ds)))
        tb_writer.add_scalar("model:tarin_loss", total_train_loss, epoch)
        tb_writer.add_scalar("model:train_acc", total_train_acc / len(train_ds), epoch)

        # save model
        ckpts_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
                    'epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': lr_scheduler_warmup.state_dict(),
                },
                ckpts_name)
        if (epoch+1) % 100 == 0:
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scheduler': lr_scheduler_warmup.state_dict(),
            }
            if args.data_type == "ESCC":
                save_name = os.path.join(save_path, "ESCC_{}.pth".format(epoch+1))
            torch.save(state, save_name)

        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scheduler': lr_scheduler_warmup.state_dict(),
            }
            save_name = os.path.join(save_path, "model_best_train.pth")
            torch.save(state, save_name)

        # Validation
        #ema.apply_shadow()#ema

        net.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            for i, (pre, post, label_pre, label_post, TCL_label) in enumerate(val_dl):
                pre = pre.to(device)
                post = post.to(device)
                labels = label_post.long().to(device)
                TCL_label = TCL_label.to(device)               
                
                logits, t_pre, t_post, t_crosspre, t_crosspost = net(pre, post)
                if isinstance(logits, tuple):
                    logits = logits[0]
                _, preds = torch.max(logits.data, 1)
                accuracy = torch.sum(preds == labels.data)
                total_val_acc += accuracy
                TCL_loss = TCL(t_pre, t_post,TCL_label)
                FCL_loss = FCL(logits, labels.view(-1))
                # FCL_loss = Ordinal_loss(logits, labels, device)
                if args.loss_mode == "fixed":
                    val_loss = Weightloss1 * TCL_loss + Weightloss2 * FCL_loss
                elif args.loss_mode == "adapted":
                    val_loss = weight_params[0] * TCL_loss + weight_params[1] * FCL_loss
                total_val_loss += val_loss
                """  """

        print("model: Val loss = {}, Val acc = {}".format(total_val_loss/ len(val_dl), total_val_acc / len(val_ds)))
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scheduler': lr_scheduler_warmup.state_dict(),
            }
            save_name = os.path.join(save_path, "model_best_loss_val.pth")
            torch.save(state, save_name)
        if total_val_acc >= best_val_acc:
            best_val_acc = total_val_acc
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scheduler': lr_scheduler_warmup.state_dict(),
            }
            save_name = os.path.join(save_path, "model_best_acc_val.pth")
            torch.save(state, save_name)

        #ema.restore() #模型保存完之后再restore
        tb_writer.add_scalar("model:val_loss", total_val_loss, epoch)
        tb_writer.add_scalar("model:val_acc", total_val_acc / len(val_ds), epoch)
        tb_writer.add_scalar("model:lr", optimizer.param_groups[0]["lr"], epoch)
        # gc.collect()
        # torch.cuda.empty_cache()

    #测试并保存结果
    tb_writer.close()


if __name__ == '__main__':
    main()
