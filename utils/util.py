import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as skm
from sklearn.mixture import GaussianMixture


#### OOD detection ####
def get_score(model, dataloader, args):
    '''
        Get score (negative output confince) of test data
    '''
    scores = []

    model.eval()
    with torch.no_grad():
        for index, (img, _) in enumerate(dataloader):
            
            img = img.to(args.device)
            logits = model(img)
            
            output = torch.softmax(logits / args.T, dim=1)
            batch_msp = torch.max(output, dim=1)[0]
            scores += list(-batch_msp.data.cpu().numpy())

    return np.array(scores)


def get_confidence(model, dataloader, args):
    '''
        Get output confince of unlabeled data
    '''
    scores = []

    model.eval()
    with torch.no_grad():
        for (inputs_u_w, _), _ in dataloader:
            
            inputs_u_w = inputs_u_w.to(args.device)
            logits = model(inputs_u_w)

            prob = torch.softmax(logits / args.T, dim=1)
            batch_msp = torch.max(prob, dim=1)[0]
            scores += list(batch_msp.data.cpu().numpy())

    return np.array(scores)


def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood):
    return np.sum(xin > np.percentile(xood, 5)) / len(xin)


#### Training ####
def get_gmm_threshold(model, dataloader, args, logger):
    '''
        Fit a two-component GMM on output confidence of unlabeled data
    '''
    confidences = get_confidence(model, dataloader, args)

    gmm = GaussianMixture(n_components=2,max_iter=100,tol=1e-2,reg_covar=5e-4)
    gmm.fit(confidences.reshape(-1,1))

    id_idx = gmm.means_.argmax()
    labels = gmm.predict(confidences.reshape(-1,1))
    labels = labels if id_idx == 1 else 1 - labels
    id_num = labels.sum()
    ood_num = (1 - labels).sum()

    scores_id = confidences[labels == 1]
    scores_ood = confidences[labels == 0]

    id_th, ood_th = np.mean(scores_id), np.mean(scores_ood)

    # avoid unstable selection caused by very low ood_th or very high id_th
    ood_limit = 1. / args.num_classes + 0.05
    id_limit = 1 - 0.05
    ood_th = max(ood_limit, ood_th)
    id_th = min(id_limit, id_th)
    
    logger.info('id_num, ood_num: %d, %d' % (id_num, ood_num))
    logger.info('th: %.6f, %.6f' % (id_th, ood_th))

    return id_th, ood_th


def softXEnt(input, target, reduction='none'):
    logprobs = F.log_softmax(input, dim=1)
    loss = -(target * logprobs).sum(1)
    if reduction == 'none':
        return loss
    else:
        return loss.sum() / input.shape[0]


def get_lam(epoch, args):
    epoch += 1
    if epoch < args.cr_warmup:
        lam_u, lam_o = args.lambda_u, 0
    else:
        lam_u, lam_o = 0, args.lambda_o

    return lam_u, lam_o


def mixup_data(x, args, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    lam = max(lam, 1 - lam)
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(args.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x



