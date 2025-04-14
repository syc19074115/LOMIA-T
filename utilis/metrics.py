import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve, confusion_matrix,matthews_corrcoef, balanced_accuracy_score
from _collections import OrderedDict

def evaluate(y_true, y_pred, digits=4, cutoff='auto'):
    '''
    calculate several metrics of predictions

    Args:
        y_true: list, labels
        y_pred: list, predictions
        digits: The number of decimals to use when rounding the number. Default is 4
        cutoff: float or 'auto'

    Returns:
        evaluation: dict

    '''

    if cutoff == 'auto':
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        youden = tpr-fpr
        cutoff = thresholds[np.argmax(youden)]

    y_pred_t = [1 if i > 0.5 else 0 for i in y_pred]
    # y_pred_t = [1 if i >= cutoff else 0 for i in y_pred]

    evaluation = OrderedDict()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
    evaluation['TP'] =tp
    evaluation['FN'] =fn
    evaluation['FP'] =fp
    evaluation['TN'] =tn
    evaluation['auc'] = round(roc_auc_score(y_true, y_pred), digits)
    evaluation['acc'] = round(accuracy_score(y_true, y_pred_t), digits)
    evaluation['recall'] = round(recall_score(y_true, y_pred_t), digits)
    evaluation['balance_acc'] = round(balanced_accuracy_score(y_true, y_pred_t), digits)
    evaluation['specificity'] = round(tn / (tn + fp), digits)
    evaluation['F1'] = round(f1_score(y_true, y_pred_t), digits)
    evaluation['mcc'] = round(matthews_corrcoef(y_true,y_pred_t),digits)
    evaluation['cutoff'] = cutoff

    return evaluation



if __name__ == '__main__':
    # out = torch.tensor([[0.0630, -0.4194], [-0.4338, 0.0729], [-0.4222, 0.0769],[0.3,0.8]])
    # pred = nn.Softmax(dim=1)(out)
    # # _,pred = torch.max(pred.data,1)
    # # print(pred)
    # true = torch.tensor([[0,1],[1,0],[0,1],[1,0]])
    # acc = get_yd(pred,true,0.5)
    # print(acc)
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0.3, 0.68, 0.35, 0.41, 0.81, 0.31])
    print(evaluate(y_true,y_pred))

