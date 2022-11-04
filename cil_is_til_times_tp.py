"""
    This is based on CIL=TIL*TP, where TP = softmax
    This code can do AUC, AUC Rec (if temperature scaling is provided manually), AUCIL, TIL, TP
    This code can also provide temperature scaling, finding optimal temperature, and entropy based CIL (and TP) prediction
"""

import sys
import os
import torch
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from copy import deepcopy


################ Necessary utils #####################
class Printer:
    def __init__(self, filename=None):
        if filename is None:
            self.filename = './result_temp.txt'
        else:
            self.filename = filename
    def print(self, *object, sep=' ', end='\n', flush=False, filename='./result_temp.txt'):
        print(*object, sep=sep, end=end, file=sys.stdout, flush=flush)

        if self.filename is not None:
            filename = self.filename
        with open(filename, 'a') as f:
            print(*object, sep=sep, end=end, file=f, flush=flush)

class AUCTracker:
    def __init__(self, n_tasks, printer):
        self.print = printer
        self.mat = np.zeros((n_tasks * 2 + 1, n_tasks * 2 + 1)) - 100
        self.n_tasks = n_tasks

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(np.concatenate([
                                                        self.mat[task_id, :task_id],
                                                        self.mat[task_id, task_id + 1:self.n_tasks]
                                                        ]))

        # # Compute forgetting
        # for i in range(task_id):
        #     self.mat[-1, i] = self.mat[i, i] - self.mat[task_id, i]

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            for i in range(task_id + 1):
                for j in range(self.n_tasks):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("{:.2f}".format(self.mat[i, -1]))
            # Print forgetting and average incremental accuracy
            for i in range(self.n_tasks):
                print("\t", end='')
            print("{:.2f}".format(self.mat[-1, -1]))
        else:
            raise NotImplementedError("Type must be 'acc'")

def compute_auc(in_scores, out_scores):
    from sklearn.metrics import roc_auc_score
    # Return auc e.g. auc=0.95
    if isinstance(in_scores, list):
        in_scores = np.concatenate(in_scores)
    if isinstance(out_scores, list):
        out_scores = np.concatenate(out_scores)

    labels = np.concatenate([np.ones_like(in_scores),
                             np.zeros_like(out_scores)])
    try:
        auc = roc_auc_score(labels, np.concatenate((in_scores, out_scores)))
    except ValueError:
        print("Input contains NaN, infinity or a value too large for dtype('float64').")
        auc = -0.99
    return auc

def auc(score_dict, task_id, auc_tracker):
    """
        AUC: AUC_ij = output values of task i's heads using i'th task data (IND)
                      vs output values of task i's head using j'th task data (OOD)
        NOTE 
    """
    in_scores = score_dict[task_id][:, task_id]

    for k, val in score_dict.items():
        if k != task_id:
            ood_scores = val[:, task_id]
            auc_value = compute_auc(in_scores, ood_scores)
            auc_tracker.update(auc_value * 100, task_id, k)

def auc_cil(score_dict, task_id, last_task_id, auc_tracker):
    """
        AUC by CIL style.
        score_dict: {data_id: np.array, size (K, T), ...},
                    where K is the sample size of data_id's data and T is the number of tasks.
                    data_id ranges from 0 to T-1
        last_task_id: last learned task id. Since it's CIL style AUC,
                      it does not make sense to compare IND score against OOD score of unlearned task network
                      Previously, in AUC(), last_task_id wasn't necessary since we don't use unlearned task network's output value
        AUC_ij = output values of task i's heads using i'th task data (IND)
                 vs output values of task j's head using i'th task data (OOD)
    """
    in_scores = score_dict[task_id][:, task_id]

    scores = score_dict[task_id]
    for k in range(last_task_id + 1):
        if k != task_id:
            ood_scores = scores[:, k]
            auc_value = compute_auc(in_scores, ood_scores)
            auc_tracker.update(auc_value * 100, task_id, k)
##############################################################


tau_til = 0.1
tau_tp  = 5

pair_list = [
    # {'dataset': 'mnist_5t', 'n_tasks': 5, 'nb_cl': 2},
    {'dataset': 'cifar10_5t', 'n_tasks': 5, 'nb_cl': 2},
    # {'dataset': 'cifar100_10t', 'n_tasks': 10, 'nb_cl': 10},
    # {'dataset': 'cifar100_20t', 'n_tasks': 20, 'nb_cl': 5},
    # {'dataset': 'timgnet_5t', 'n_tasks': 5, 'nb_cl': 40},
    # {'dataset': 'timgnet_10t', 'n_tasks': 10, 'nb_cl': 20},
]
methods = [
# 'owm',
# 'muc',
# 'pass',
# 'lwf',
# 'icarl',
# 'mnemonics',
# 'bic',
# 'derpp',
# 'hat',
# 'hypernet',
# 'sup',
'sup_csi',
# 'sup_cal',
# 'hat_csi',
# 'hat_csi_cal',
]

class ComputeEnt:
    def __init__(self, temp):
        # T is for smoothing
        self.temp = temp.reshape(1, -1, 1)

    def compute(self, output, keepdims=True):
        """
            output: torch.tensor logit, 2d
        """
        soft = softmax(output / self.temp, axis=-1)
        if keepdims:
            return -1 * np.sum(soft * np.log(soft), axis=-1, keepdims=True)
        else:
            return -1 * np.sum(soft * np.log(soft))

def compute_auc(in_scores, out_scores):
    from sklearn.metrics import roc_auc_score
    # Return auc e.g. auc=0.95
    if isinstance(in_scores, list):
        in_scores = np.concatenate(in_scores)
    if isinstance(out_scores, list):
        out_scores = np.concatenate(out_scores)

    labels = np.concatenate([np.ones_like(in_scores),
                             np.zeros_like(out_scores)])
    try:
        auc = roc_auc_score(labels, np.concatenate((in_scores, out_scores)))
    except ValueError:
        print("Input contains NaN, infinity or a value too large for dtype('float64').")
        auc = -0.99
    return auc

def auc(score_dict, task_id, auc_tracker):
    """
        AUC: AUC_ij = output values of task i's heads using i'th task data (IND)
                      vs output values of task i's head using j'th task data (OOD)
        NOTE 
    """
    in_scores = score_dict[task_id][:, task_id]

    for k, val in score_dict.items():
        if k != task_id:
            ood_scores = val[:, task_id]
            auc_value = compute_auc(in_scores, ood_scores)
            auc_tracker.update(auc_value * 100, task_id, k)

def auc_cil(score_dict, task_id, last_task_id, auc_tracker):
    """
        AUC by CIL style.
        score_dict: {data_id: np.array, size (K, T), ...},
                    where K is the sample size of data_id's data and T is the number of tasks.
                    data_id ranges from 0 to T-1
        last_task_id: last learned task id. Since it's CIL style AUC,
                      it does not make sense to compare IND score against OOD score of unlearned task network
                      Previously, in AUC(), last_task_id wasn't necessary since we don't use unlearned task network's output value
        AUC_ij = output values of task i's heads using i'th task data (IND)
                 vs output values of task j's head using i'th task data (OOD)
    """
    in_scores = score_dict[task_id][:, task_id]

    scores = score_dict[task_id]
    for k in range(last_task_id + 1):
        if k != task_id:
            ood_scores = scores[:, k]
            auc_value = compute_auc(in_scores, ood_scores)
            auc_tracker.update(auc_value * 100, task_id, k)

def compute_til_prob(nb_cl, outputType, output, epsilon):
    """
        Convert output to TIL probability
        args: output, shape (B, C)
        return: prob, shape (B, num_task_learned, nb_cl)
    """
    B, C = output.shape

    if epsilon is not None:
        noise = np.random.normal(scale=epsilon, size=(B, C))
        output = output + noise

    if outputType == '':
        output = softmax(output.reshape(B, -1, nb_cl) / tau_til, axis=-1)

    if output.ndim != 3:
        output = output.reshape(B, -1 , nb_cl)

    return output

def compute_tp_prob(nb_cl, outputType, output, epsilon, preTemp):
    """
        TP = softmax(max(softmax(task1)), ..., max(softmax(task_n)))
        In other word, TP is based on OOD, as OOD_i = max(softmax(task_i))
        Return: tp_prob; shape (B, num_tasks_learned, -1), scores; shape (B, num_tasks_learned)
    """
    B, C = output.shape
    nb_t = C // nb_cl

    if preTemp is not None:
        output = output.reshape(B, -1, nb_cl) / preTemp.reshape(1, -1, 1)
        output = output.reshape(B, -1)

    if epsilon is not None:
        noise = np.random.normal(scale=epsilon, size=(B, C))
        output = output + noise

    scores = np.max(output.reshape(B, -1, nb_cl), axis=2)
    output = output.reshape(B, -1, nb_cl)
    prob = np.max(softmax(output / tau_tp, axis=-1), axis=2)
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    
    output = output.reshape(B, -1)
    prob = np.expand_dims(prob, -1)
    return prob, scores

def optimal_temp(n_tasks, nb_cl, outputType, output1, targets, preTemp):
    if preTemp is None:
        preTemp = torch.ones(n_tasks) * 20

    if isinstance(preTemp, np.ndarray):
        preTemp = torch.from_numpy(preTemp)

    from torch.optim import SGD
    preTemp = preTemp.view(1, -1, 1)
    preTemp.requires_grad = True
    optim = SGD([preTemp], lr=0.00001, momentum=0.9)

    output1 = torch.from_numpy(output1)
    targets = torch.from_numpy(targets)

    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn
    import torch.nn.functional as F

    dataset = TensorDataset(output1, targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for epoch in range(4):
        loss_total = 0
        for x, y in loader:
            B, C = x.size()
            out = x.reshape(B, -1, nb_cl) * preTemp
            out = F.softmax(out, dim=-1).view(B, -1)

            loss = nn.CrossEntropyLoss()(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_total += loss.item()
        print(loss_total / len(loader))

    preTemp = preTemp.view(-1)
    print(preTemp)
    return preTemp.data.numpy()

def compute_accs(nb_cl, data_id, outputType, output1, output2, targets, epsilon=None, preTemp=None, entropy=False):
    """
        Compute CIL acc, TIL acc, and TP acc
        Return CIL_acc, TIL_acc, TP_acc, scores
    """
    B, C = output1.shape

    if preTemp is not None:
        assert preTemp.ndim == 1

    til_prob = compute_til_prob(nb_cl, outputType, output1, epsilon)
    tp_prob, scores = compute_tp_prob(nb_cl, outputType, output2, epsilon, preTemp)
    if entropy is not None:
        ent_val = entropy.compute(output1.reshape(B, -1, nb_cl))
        ent_val = ent_val.reshape(B, -1)
        # ent_val = np.sum(ent_val, 0)
        tp_prob = softmax(-1 * ent_val, -1)
        task_pred = np.argmin(ent_val, -1)
        task_pred_correct = task_pred == data_id

    # CIL acc
    cil_prob = (til_prob * tp_prob).reshape(B, -1)
    pred = np.argmax(cil_prob, axis=1)
    # if preTemp is not None:
    #     output1 = output1.reshape(B, -1, nb_cl)
    #     output1 = output1 / preTemp.reshape(1, -1, 1)
    #     output1 = softmax(output1, axis=-1).reshape(B, -1)
    # pred = np.argmax(output1, axis=1)

    if entropy is None:
        cil_acc = sum(pred == targets) / len(targets) * 100
    else:
        cil_acc = sum(pred[task_pred_correct] == targets[task_pred_correct]) / len(targets) * 100

    # TIL acc
    til_prob = til_prob[:, data_id]
    pred = np.argmax(til_prob, axis=1)
    normalized_targets = targets % nb_cl
    til_acc = sum(pred == normalized_targets) / len(targets) * 100

    # TP acc
    pred = np.argmax(tp_prob.reshape(B, -1), axis=1)
    tp_acc = sum(pred == data_id) / len(targets) * 100

    return cil_acc, til_acc, tp_acc, scores


cil_table, auc_table, std_cil_table, std_auc_table = [], [], [], []
printer = Printer().print
for pair in pair_list:
    preTemp = None
    # preTemp = np.array([1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    # preTemp = np.array([50, 50, 45, 45, 45, 45, 45, 45, 45, 45])
    # preTemp = np.array([20, 20, 20, 20, 20])
    find_optimal_temp = False
    # preTemp = np.array([2.1, 0.81, 0.72, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    # preTemp = np.ones(pair['n_tasks']) * 0.95
    entropy = None
    # entropy = ComputeEnt(preTemp)

    if entropy is not None:
        preTemp = None

    plt.figure(figsize=(7, 6))
    dataset = pair['dataset']
    net_id = pair['n_tasks'] - 1
    data_id = net_id # only consider the last results
    n_tasks = pair['n_tasks']
    for method in methods: # muc, pass, etc.
        results = os.listdir(f'./logs/{dataset}_csi')

        list_of_exp = []
        for exp in results: # e.g. exp = 'cifar100_20t_round=0'
            if dataset in exp and os.path.isdir(f'./logs/{dataset}_csi/{exp}'):
                if 'round' in exp:
                    list_of_exp.append(exp)

        print(list_of_exp)
        # list_of_exp = [list_of_exp[0]]

        def loader(type):
            # type choices: '', 'temp', 'msp', 'odin'
            if type == '':
                arr = torch.load(f'./logs/{dataset}_csi/{exp}/outputs_labels_list_{net_id}')
            elif type == 'odin':
                arr = torch.load(f'./logs/{dataset}_csi/{exp}/outputs_labels_temp_noise_list_{net_id}')
            else:
                arr = torch.load(f'./logs/{dataset}_csi/{exp}/outputs_labels_{type}_list_{net_id}')
            return arr

        # This includes all the round for the experiment e.g. [cifar10_round=0, ..., cifar10_round=4]
        acc_of_all_types_rounds = []
        auc_of_all_types_rounds = []
        auc_of_all_types_rounds_recomputed = []
        auc_cil_of_all_types_rounds = []
        acc_of_all_types_til_rounds = []
        acc_of_all_types_tp_rounds = []

        preTemp_copy = preTemp
        for exp in list_of_exp: # run through r many times
            arr = loader(type='')
            # arr_msp = loader(type='msp')
            # arr_temp = loader(type='temp')
            # arr_odin = loader(type='odin')


            acc_of_all_types, auc_of_all_types, auc_cil_of_all_types, acc_of_all_types_til, acc_of_all_types_tp = [], [], [], [], []
            auc_softmax_tracker = AUCTracker(pair['n_tasks'], printer=printer)
            auc_of_all_types_recomputed = []
            # for type_name in ['', 'msp', 'temp', 'odin']:
            for type_name in ['']:
                new_acc, new_acc_til, new_acc_tp = [], [], []
                score_dict = {}
                new_arr = loader(type=type_name)

                data_all, target_all = [], []
                if find_optimal_temp:
                    for data_id in range(n_tasks):
                        data_all.append(new_arr[0][data_id])
                        target_all.append(new_arr[1][data_id])
                    data_all = np.concatenate(data_all)
                    target_all = np.concatenate(target_all)

                    preTemp = optimal_temp(n_tasks=n_tasks, nb_cl=pair['nb_cl'], outputType=type_name,
                                 output1=data_all, targets=target_all, preTemp=preTemp_copy)

                for data_id in range(n_tasks):
                    # n_tasks, nb_cl, data_id, outputs1, outputs2, targets
                    # acc = cil_acc(output1=new_arr[0][data_id], output2=new_arr[0][data_id], targets=new_arr[1][data_id])
                    acc, acc_til, acc_tp, scores = compute_accs(nb_cl=pair['nb_cl'], data_id=data_id, outputType=type_name,
                                output1=new_arr[0][data_id], output2=new_arr[0][data_id], targets=new_arr[1][data_id],
                                preTemp=preTemp, entropy=entropy)
                    score_dict[data_id] = scores
                    # acc, acc_til, acc_tp = compute_acc_by_pairs(n_tasks, pair['nb_cl'], data_id, new_arr[0][data_id], new_arr[0][data_id], new_arr[1][data_id])
                    # print(exp, type_name, acc)
                    new_acc.append(acc)
                    new_acc_til.append(acc_til)
                    new_acc_tp.append(acc_tp)

                for data_id in range(n_tasks):
                    auc(score_dict, data_id, auc_softmax_tracker)
                auc_of_all_types_recomputed.append(auc_softmax_tracker.mat[-1, -1])

                avg_acc = np.mean(new_acc)
                avg_acc_til = np.mean(new_acc_til)
                avg_acc_tp = np.mean(new_acc_tp)

                acc_of_all_types.append(avg_acc)
                acc_of_all_types_til.append(avg_acc_til)
                acc_of_all_types_tp.append(avg_acc_tp)

            # acc_of_all_types is now a list with 4 accs
            acc_of_all_types_rounds.append(np.array(acc_of_all_types).reshape(1, -1))
            auc_of_all_types_rounds.append(np.array(auc_of_all_types).reshape(1, -1))
            auc_of_all_types_rounds_recomputed.append(np.array(auc_of_all_types_recomputed).reshape(1, -1))
            auc_cil_of_all_types_rounds.append(np.array(auc_cil_of_all_types).reshape(1, -1))
            acc_of_all_types_til_rounds.append(np.array(acc_of_all_types_til).reshape(1, -1))
            acc_of_all_types_tp_rounds.append(np.array(acc_of_all_types_tp).reshape(1, -1))

        acc_of_all_types_rounds = np.concatenate(acc_of_all_types_rounds, 0)
        avg_acc_over_rounds = np.mean(np.array(acc_of_all_types_rounds), axis=0)
        std_acc_over_rounds = np.std(np.array(acc_of_all_types_rounds), axis=0, ddof=1)

        auc_of_all_types_rounds = np.concatenate(auc_of_all_types_rounds, 0)
        avg_auc_over_rounds = np.mean(np.array(auc_of_all_types_rounds), axis=0)
        std_auc_over_rounds = np.std(np.array(auc_of_all_types_rounds), axis=0, ddof=1)

        auc_of_all_types_rounds_recomputed = np.concatenate(auc_of_all_types_rounds_recomputed, 0)
        avg_auc_over_rounds_recomputed = np.mean(np.array(auc_of_all_types_rounds_recomputed), axis=0)
        std_auc_over_rounds_recomputed = np.std(np.array(auc_of_all_types_rounds_recomputed), axis=0, ddof=1)

        auc_cil_of_all_types_rounds = np.concatenate(auc_cil_of_all_types_rounds, 0)
        avg_auc_cil_over_rounds = np.mean(np.array(auc_cil_of_all_types_rounds), axis=0)
        std_auc_cil_over_rounds = np.std(np.array(auc_cil_of_all_types_rounds), axis=0, ddof=1)

        acc_of_all_types_til_rounds = np.concatenate(acc_of_all_types_til_rounds, 0)
        avg_til_over_rounds = np.mean(np.array(acc_of_all_types_til_rounds), axis=0)
        std_til_over_rounds = np.std(np.array(acc_of_all_types_til_rounds), axis=0, ddof=1)

        acc_of_all_types_tp_rounds = np.concatenate(acc_of_all_types_tp_rounds, 0)
        avg_tp_over_rounds = np.mean(np.array(acc_of_all_types_tp_rounds), axis=0)
        std_tp_over_rounds = np.std(np.array(acc_of_all_types_tp_rounds), axis=0, ddof=1)
        
        print(method, "AUC\t\t", avg_auc_over_rounds)
        print(method, "AUC Rec\t", avg_auc_over_rounds_recomputed)
        print(method, "AUC CIL\t", avg_auc_cil_over_rounds)
        print(method, "ACC\t\t", avg_acc_over_rounds)
        print(method, "TIL\t\t", avg_til_over_rounds)
        print(method, "TP\t\t", avg_tp_over_rounds)

        cil_table.append(avg_acc_over_rounds[0])
        # auc_table.append(avg_auc_over_rounds[0])
        std_cil_table.append(std_acc_over_rounds[0])
        # std_auc_table.append(std_auc_over_rounds[0])
        auc_table = cil_table
        std_auc_table = std_cil_table

last = len(auc_table)
for i, (auc, std_auc, cil, std_cil) in enumerate(zip(auc_table, std_auc_table, cil_table, std_cil_table)):
    i += 1
    if i == last:
        print("{:.2f} & {:.2f} \\\\".format(auc, cil))
    else:
        print("{:.2f} & {:.2f} & ".format(auc, cil), end='')

print()
last = len(auc_table)
for i, (auc, std_auc, cil, std_cil) in enumerate(zip(auc_table, std_auc_table, cil_table, std_cil_table)):
    i += 1
    if i == last:
        # print("{:.1f}".format(auc), end='')
        # print("\scalebox{1.0}")
        print("{:.1f}\scalebox{{1.0}}{{$\pm${:.2f}}} \\\\".format(cil, std_cil))
    else:
        # print("{:.1f}".format(auc), end='')
        # print("\scalebox{1.0}{$\pm$", end='')
        # print("{:.2f}}".format(std_auc))
        print("{:.1f}\scalebox{{1.0}}{{$\pm${:.2f}}} & ".format(cil, std_cil), end='')