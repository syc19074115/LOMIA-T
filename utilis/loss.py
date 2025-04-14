# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        # print("num",class_num)
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print(probs.shape, log_p.shape)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print("batch_loss",batch_loss.shape)
        # print('-----bacth_loss------')
        # print("batch_loss",batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
            # print("loss",loss)
        else:
            loss = batch_loss.sum()
        return loss
    
class ContrastiveLoss_euc(nn.Module):  #计算欧氏距离
    def __init__(self, margin=2.0):
        super(ContrastiveLoss_euc, self).__init__()
        self.margin = margin
        #self.batch = batch_size
        
    def forward(self, output1, output2, label):  #具有差异性，label为1；不具有差异性，label为0  label为1时，希望欧氏距离更小
        output1 = torch.mean(output1, dim=1)
        output2 = torch.mean(output2, dim=1)
        euclidean_distance = F.pairwise_distance(output1, output2).reshape(-1,1)
        # print(euclidean_distance, label)
        # for i in range(len(label)):
        #     print(euclidean_distance[i], label[i])
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
class ContrastiveLoss_euc_siamese(nn.Module):  #计算欧氏距离
    def __init__(self, margin=2.0):
        super(ContrastiveLoss_euc_siamese, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):  #具有差异性，label为0；不具有差异性，label为1  label为0时，希望欧氏距离更小
        euclidean_distance = F.pairwise_distance(output1, output2)
        #print(euclidean_distance)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        #loss_contrastive = torch.mean(part1 + part2)
        #print(torch.mean(loss_contrastive))
        return loss_contrastive

if __name__ == '__main__':
    input1 = torch.randn([32,9,32])
    input2 = torch.randn([32,9,32])
    label = torch.randint(0, 2, [32])
    print(label)
    criterion = ContrastiveLoss_euc(margin=2)
    loss = criterion(input1, input2, label)
    print(loss)

    """ criterion = SupConLoss(contrast_mode='one')
    #pre_out = input1.reshape([32,9,32])
    #post_out = input2.reshape([32,9,32])

    features = torch.stack((input1,input2),dim=1)
    print(features.shape)
    loss = criterion(features, label, None)
    print(loss) """

    """ triplet_loss = \
    nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    output = triplet_loss(label, positive, negative) """
    