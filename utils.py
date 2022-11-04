import os, sys

from PIL import Image
import time
import pathlib

import torch
import torch.nn as nn
import models
import models.module_util as module_util
import torch.backends.cudnn as cudnn

from models.modules import FastMultitaskMaskConv, MultitaskMaskConv
from args import args

import numpy as np
from datetime import datetime

def cond_cache_masks(m,):
    if hasattr(m, "cache_masks"):
        m.cache_masks()


def cond_cache_weights(m, t):
    if hasattr(m, "cache_weights"):
        m.cache_weights(t)


def cond_clear_masks(m,):
    if hasattr(m, "clear_masks"):
        m.clear_masks()


def cond_set_mask(m, task):
    if hasattr(m, "set_mask"):
        m.set_mask(task)


def cache_masks(model):
    model.apply(cond_cache_masks)


def cache_weights(model, task):
    model.apply(lambda m: cond_cache_weights(m, task))


def clear_masks(model):
    model.apply(cond_clear_masks)


def set_mask(model, task):
    model.apply(lambda m: cond_set_mask(m, task))


def freeze_model_weights(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Freezing weight for {n}")
            m.weight.requires_grad_(False)

            if m.weight.grad is not None:
                m.weight.grad = None
                print(f"==> Resetting grad value for {n} -> None")


def freeze_model_scores(model: nn.Module, task_idx: int):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Freezing weight for {n}")
            m.scores[task_idx].requires_grad_(False)

            if m.scores[task_idx].grad is not None:
                m.scores[task_idx].grad = None
                print(f"==> Resetting grad value for {n} scores -> None")


def unfreeze_model_weights(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Unfreezing weight for {n}")
            m.weight.requires_grad_(True)


def unfreeze_model_scores(model: nn.Module, task_idx: int):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Unfreezing weight for {n}")
            m.scores[task_idx].requires_grad_(True)


def set_gpu(model):
    if args.multigpu is None:
        args.device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )
        args.device = torch.cuda.current_device()
        cudnn.benchmark = True

    return model


def get_model():
    model = models.__dict__[args.model]()
    return model


def write_result_to_csv(**kwargs):
    results = pathlib.Path(args.log_dir) / "results.csv"

    if not results.exists():
        results.write_text("Date Finished,Name,Current Val,Best Val,Save Directory\n")

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{name}, "
                "{curr_acc1:.04f}, "
                "{best_acc1:.04f}, "
                "{save_dir}\n"
            ).format(now=now, **kwargs)
        )


def write_adapt_results(**kwargs):
    results = pathlib.Path(args.run_base_dir) / "adapt_results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished,"
            "Name,"
            "Task,"
            "Num Tasks Learned,"
            "Current Val,"
            "Adapt Val\n"
        )
    now = time.strftime("%m-%d-%y_%H:%M:%S")
    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{name}~task={task}~numtaskslearned={num_tasks_learned}~tasknumber={task_number}, "
                "{task}, "
                "{num_tasks_learned}, "
                "{curr_acc1:.04f}, "
                "{adapt_acc1:.04f}\n"
            ).format(now=now, **kwargs)
        )


class BasicVisionDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform, target_transform):
        assert len(data) == len(targets)

        self.data = data
        self.targets = targets

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def kth_elt(x, base):
    if base == 2:
        return x.median()
    else:
        val, _ = x.flatten().sort()
        return val[(val.size(0) - 1) // base]

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix

def Supervised_NT_xent(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    if multi_gpu:
        gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        labels = torch.cat(distops.all_gather(gather_t, labels))
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    #Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss

class Logger:
    def __init__(self, args, name=None):
        self.init = datetime.now()
        self.args = args
        if name is None:
            self.name = self.init.strftime("%m|%d|%Y %H|%M|%S")
        else:
            self.name = name

        self.args.dir = self.name

        self._make_dir()

    def now(self):
        time = datetime.now()
        diff = time - self.init
        self.print(time.strftime("%m|%d|%Y %H|%M|%S"), f" | Total: {diff}")

    def print(self, *object, sep=' ', end='\n', flush=False, filename='/result.txt'):
        print(*object, sep=sep, end=end, file=sys.stdout, flush=flush)

        if self.args.print_filename is not None:
            filename = self.args.print_filename
        with open(self.dir() + filename, 'a') as f:
            print(*object, sep=sep, end=end, file=f, flush=flush)

    def _make_dir(self):
        # If provided hdd drive
        if 'hdd' in self.name or 'sdb' in self.name:
            if not os.path.isdir('/' + self.name):
                os.makedirs('/' + self.name)
        else:
            if not os.path.isdir(self.name):
                os.makedirs(self.name)
            # if not os.path.isdir(''):
            #     os.mkdir('')

    def dir(self):
        if 'hdd' in self.name or 'sdb' in self.name:
            return '/' + self.name + '/'
        else:
            return f'./{self.name}/'
            # './logs/{}/'.format(self.name)

    def time_interval(self):
        self.print("Total time spent: {}".format(datetime.now() - self.init))

class Tracker:
    def __init__(self, args):
        self.print = args.logger.print
        self.mat = np.zeros((args.n_tasks * 2 + 1, args.n_tasks * 2 + 1)) - 100

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(self.mat[task_id, :p_task_id + 1])

        # Compute forgetting
        for i in range(task_id):
            self.mat[-1, i] = self.mat[i, i] - self.mat[task_id, i]

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            for i in range(task_id + 1):
                for j in range(task_id + 1):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("{:.2f}".format(self.mat[i, -1]))
        elif type == 'forget':
            # Print forgetting and average incremental accuracy
            for i in range(task_id + 1):
                acc = self.mat[-1, i]
                if acc != -100:
                    print("{:.2f}\t".format(acc), end='')
                else:
                    print("\t", end='')
            print("{:.2f}".format(self.mat[-1, -1]))
            if task_id > 0:
                forget = np.mean(self.mat[-1, :task_id])
                print("{:.2f}".format(forget))
        else:
            raise NotImplementedError("Type must be either 'acc' or 'forget'")

class AUCTracker:
    def __init__(self, args):
        self.print = args.logger.print
        self.mat = np.zeros((args.n_tasks * 2 + 1, args.n_tasks * 2 + 1)) - 100
        self.n_tasks = args.n_tasks

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

class AUCILTracker(AUCTracker):
    def __init__(self, args):
        super(AUCILTracker, self).__init__(args)
        self.last_id = 0

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.last_id = max([self.last_id, p_task_id])

        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(np.concatenate([
                                                        self.mat[task_id, :task_id],
                                                        self.mat[task_id, task_id + 1:self.last_id + 1]
                                                        ]))

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

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
