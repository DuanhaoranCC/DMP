import numpy as np
import torch
# from utils.logreg import LogReg
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import functools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score


class LogReg(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)
        # return torch.log_softmax(self.fc(seq).squeeze(), dim=-1)


class EvaData(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.y = y

    def __getitem__(self, item):
        return self.data[item], self.y[item]

    def __len__(self):
        return len(self.data)


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, train_mask, val_mask, test_mask, label, device, data, lr, wd, name):
    num_features = embeds.shape[1]
    num_classes = label.max() + 1
    xent = nn.CrossEntropyLoss()
    embeds = embeds.to(device)
    label = label.to(device)
    if name == 'aminer':
        embeds = embeds[data.label[:, 0]]
    if name == 'cite':
        num_classes = label.size(1)
        xent = nn.BCEWithLogitsLoss()

    if name == 'cs':
        train_lbls = label[torch.nonzero(train_mask)].long().squeeze()
        val_lbls = label[torch.nonzero(val_mask)].long().squeeze()
        test_lbls = label[torch.nonzero(test_mask)].long().squeeze()
        # Field
        # num_classes = label.size(1)
        # xent = nn.KLDivLoss(reduction='batchmean')
        # # xent = nn.BCEWithLogitsLoss()
        # train_lbls = label[torch.nonzero(train_mask)].float().squeeze()
        # # val_lbls = label[torch.nonzero(val_mask)].float().squeeze()
        # test_lbls = label[torch.nonzero(test_mask)].float().squeeze()
        # train_embs = embeds[train_mask]
        # test_embs = embeds[test_mask].to(device)
        # dataset = EvaData(train_embs, train_lbls)
        # da = DataLoader(dataset, batch_size=50000, shuffle=True, num_workers=0)
    else:
        train_lbls = label[train_mask]
        val_lbls = label[val_mask]
        test_lbls = label[test_mask]
    train_embs = embeds[train_mask]
    val_embs = embeds[val_mask]
    test_embs = embeds[test_mask]

    if name == 'cite':
        train_lbls = train_lbls.float()
        # dataset = EvaData(train_embs, train_lbls)
        # da = DataLoader(dataset, batch_size=50000, shuffle=True, num_workers=0)
        val_lbls = val_lbls.float()
        test_lbls = test_lbls.float()

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(1):
        log = LogReg(num_features, num_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        logits_list = []
        # train_embs = train_embs.to(device)
        # train_lbls = train_lbls.to(device)
        for i in range(200):
            # train
            # for index, (train_x, train_y) in enumerate(da):
            #     log.train()
            #     opt.zero_grad()
            #     logits = log(train_x.to(device))
            #     loss = xent(logits, train_y.to(device))
            #     loss.backward()
            #     opt.step()

            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            # print(i, loss.item())
            # preds = torch.argmax(logits, dim=1)
            # train_acc = torch.sum(preds == train_lbls).float() / train_lbls.shape[0]
            loss.backward()
            opt.step()
            ##########################################################################
            # logits = log(val_embs.to(device))
            # #####################################################################
            # Paper-Venue
            # valid_res = []
            # for ai, bi in zip(val_lbls, logits.argsort(descending=True)):
            #     valid_res += [(bi == ai).int().tolist()]
            # valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
            # val_micro_f1s.append(valid_ndcg)
            #####################################################################
            # Paper-Field
            # valid_res = []
            # for ai, bi in zip(val_lbls, logits.argsort(descending=True)):
            #     valid_res += [ai[bi.cpu().numpy()].cpu().numpy()]
            # valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
            # val_micro_f1s.append(valid_ndcg)
            #####################################################################
            # if name == 'cite':
            #     preds = (logits > 0).float()
            # else:
            #     preds = torch.argmax(logits, dim=1)
            # # val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            # val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            # val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')
            #
            # # val_accs.append(val_acc.item())
            # val_macro_f1s.append(val_f1_macro)
            # val_micro_f1s.append(val_f1_micro)
            #
        # Test
        logits = log(test_embs)
        # #################################################################################
        # Paper-Venue
        test_res = []
        for ai, bi in zip(test_lbls, logits.argsort(descending=True)):
            test_res += [(bi == ai).int().tolist()]
        test_mrr = np.average(mean_reciprocal_rank(test_res))
        test_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in test_res])
        micro_f1s.append(test_ndcg)
        macro_f1s.append(test_mrr)
        print("Test: ", test_mrr, test_ndcg)
        #################################################################################
        # Paper-Field
        # test_res = []
        # for ai, bi in zip(test_lbls, logits.argsort(descending=True)):
        #     test_res += [ai[bi.cpu().numpy()].cpu().numpy()]
        # test_mrr = np.average(mean_reciprocal_rank(test_res))
        # test_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in test_res])
        # # test_micro_f1s.append(test_ndcg)
        # print("Test: ", test_mrr, test_ndcg)
        #################################################################################
        # if name == 'cite':
        #     preds = (logits > 0).float()
        # else:
        #     preds = torch.argmax(logits, dim=1)
        # # test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        # test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
        # test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

        # print(train_acc, val_acc, test_acc)
        # test_accs.append(test_acc.item())
        # test_macro_f1s.append(test_f1_macro)
        # test_micro_f1s.append(test_f1_micro)
        # logits_list.append(logits)
        #################################################################################

        # max_iter = val_accs.index(max(val_accs))
        # accs.append(test_accs[max_iter])
        #################################################################################
        # max_iter = val_macro_f1s.index(max(val_macro_f1s))
        # macro_f1s.append(test_macro_f1s[max_iter])
        # macro_f1s_val.append(val_macro_f1s[max_iter])
        #################################################################################
        # max_iter = val_micro_f1s.index(max(val_micro_f1s))
        # micro_f1s.append(test_micro_f1s[max_iter])
        # auc
        # if name != 'cite':
        #     best_logits = logits_list[max_iter]
        #     best_proba = softmax(best_logits, dim=1)
        #     auc_score_list.append(
        #         roc_auc_score(
        #             y_true=F.one_hot(test_lbls).detach().cpu().numpy(),
        #             y_score=best_proba.detach().cpu().numpy(),
        #             multi_class='ovr'
        #         )
        #     )

    # print("Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f} var: {:.4f}"
    #     .format(
    #     np.mean(macro_f1s),
    #     np.std(macro_f1s),
    #     np.mean(micro_f1s),
    #     np.std(micro_f1s),
    #     np.mean(auc_score_list),
    #     np.std(auc_score_list)
    # )
    # )
    # np.savetxt('./ndcg.txt', np.array(micro_f1s), fmt="%.4f")
    # np.savetxt('./mrr.txt', np.array(macro_f1s), fmt="%.4f")
    return test_ndcg
    # return np.mean(micro_f1s)


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret
