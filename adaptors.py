from args import args
import torch
import torch.nn as nn
from torch import optim
import math

import numpy as np
import pathlib

from models.modules import FastHopMaskBN
from models import module_util
from utils import kth_elt
from functools import partial

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def adapt_test(
    model,
    test_loader,
    alphas=None,
    til=False
):
    correct = 0
    model.eval()
    num_cls = args.output_size
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target % num_cls
            if alphas is not None:
                model.apply(lambda m: setattr(m, "alphas", alphas))
            
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = float(correct) / len(test_loader.dataset)

    model.train()
    return test_acc

def auc(model, id_loader, ood_loaders, num_tasks_learned, task, temperature=None):
    model.eval()
    def score_func(x, temperature):
        total_outputs = model(x)
        if temperature is None: temperature = 1

        scores = F.softmax(total_outputs / temperature, dim=1).max(dim=1)[0]
        return scores

    def get_scores(loader, temperature):
        scores = []
        for x, _ in loader:
            s = score_func(x, temperature)
            scores.append(s.detach().cpu().numpy())
        return np.concatenate(scores)

    def get_auroc(scores_id, scores_ood):
        scores = np.concatenate([scores_id, scores_ood])
        labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
        return roc_auc_score(labels, scores)

    alphas = (
        torch.zeros(
            [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=False
        )
    )
    alphas[task] = 1
    alphas.requires_grad = True

    model.apply(lambda m: setattr(m, "alphas", alphas))
    model.apply(lambda m: setattr(m, "task", task))

    if alphas is not None:
        model.apply(lambda m: setattr(m, "alphas", alphas))

    auroc_dict = {}
    scores_id = get_scores(id_loader, temperature)

    for ood, ood_loader in ood_loaders.items():
        scores_ood = get_scores(ood_loader, temperature)
        auroc_dict[ood] = get_auroc(scores_id, scores_ood)

    model.apply(lambda m: setattr(m, "alphas", None))
    model.train()
    return auroc_dict

def adapt_test_cil(model, test_loader, num_tasks_learned, task, temperature=None, noise=None):
    """
        num_tasks_learned: last learned task id + 1
        task: is the correct data id for the loader
    """
    model.eval()

    num_cls = args.output_size
    cil_correct, correct_task_pred, total = 0, 0, 0
    score_list_all = []
    output_list, label_list = [], []
    for batch_idx, (data, target) in enumerate(test_loader):
        target = target % num_cls
        data, target = data.to(args.device), target.to(args.device)

        cil_output_list = torch.tensor([]).to(args.device)
        score_list_task = []
        for tt in range(num_tasks_learned):
            alphas = (
                torch.zeros(
                    [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=False
                )
            )
            alphas[tt] = 1
            alphas.requires_grad = True
        
            model.apply(lambda m: setattr(m, "alphas", alphas))
            model.apply(lambda m: setattr(m, "task", tt))

            if alphas is not None:
                model.apply(lambda m: setattr(m, "alphas", alphas))

            # Added: Start perturbation
            if noise is not None:
                data.requires_grad = True
            # Added: End perturbation

            output = model(data)

            # Added: Start perturbation
            if noise is not None:
                output = output / temperature
                target_aux = output.argmax(1).data

                loss = nn.CrossEntropyLoss()(output, target_aux)
                loss.backward()

                gradient = torch.ge(data.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2
                gradient[:, 0] = (gradient[:, 0]) / args.std[0] # these are std for each channel
                gradient[:, 1] = (gradient[:, 1]) / args.std[1]
                gradient[:, 2] = (gradient[:, 2]) / args.std[2]
                tempInputs = torch.add(data.data, -noise, gradient)

                with torch.no_grad():
                    output = model(tempInputs)
            # Added: End perturbation

            if temperature is not None:
                output = F.softmax(output.data / temperature)

            cil_output_list = torch.cat((cil_output_list, output.data), dim=1)

            scores, _ = output.data.max(1, keepdim=True)
            score_list_task.append(scores)

        score_list_task = torch.cat(score_list_task, dim=1)
        score_list_all.append(score_list_task)

        target = num_cls * task + target
        cil_correct += cil_output_list.argmax(1).eq(target).sum().item()

        task_pred = cil_output_list.argmax(1) // num_cls
        task_labels = torch.zeros_like(target).to(args.device) + task
        correct_task_pred += task_pred.eq(task_labels).sum().item()

        output_list.append(cil_output_list.data.cpu().numpy())
        label_list.append(target.data.cpu().numpy())

        total += len(target)

    acc = cil_correct / total * 100
    task_acc = correct_task_pred / total * 100

    model.apply(lambda m: setattr(m, "alphas", None))

    score_list_all = torch.cat(score_list_all)
    output_list = np.concatenate(output_list)
    label_list = np.concatenate(label_list)
    model.train()
    return acc, task_acc, score_list_all, output_list, label_list

def adapt_test_csi(
    model,
    test_loader,
    alphas=None,
    til=False
):
    # TIL
    num_cls = args.output_size
    correct = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if alphas is not None:
                model.apply(lambda m: setattr(m, "alphas", alphas))
            
            data, target = data.to(args.device), target.to(args.device)
            output = 0
            for i in range(4):
                rot_data = torch.rot90(data, i, (2, 3))
                _, outputs_aux = model(rot_data, joint=True)
                output += outputs_aux['joint'][:, num_cls * i:num_cls * (i + 1)] / 4.
            # output = model(data)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = float(correct) / len(test_loader.dataset)

        # args.logger.print(
        #     f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
        # )
    model.train()
    return test_acc

def adapt_test_cil_csi(model, test_loader, num_tasks_learned, task, w=None, b=None, temperature=None, noise=None):
    # CIL
    if args.cal_pretrain is not None:
        cal_pretrain = torch.load(args.cal_pretrain)
        w = cal_pretrain['w'].to(args.device)
        b = cal_pretrain['b'].to(args.device)
    if w is not None:
        args.logger.print(w)
        args.logger.print(b)

    num_cls = args.output_size
    output_list, label_list = [], []
    model.eval()

    with torch.no_grad():
        cil_correct, correct_task_pred, total = 0, 0, 0
        score_list_all = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)

            cil_output_list = torch.tensor([]).to(args.device)
            score_list_task = []
            for tt in range(num_tasks_learned):
                alphas = (
                    torch.zeros(
                        [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=False
                    )
                )
                alphas[tt] = 1
                alphas.requires_grad = True
            
                model.apply(lambda m: setattr(m, "alphas", alphas))
                model.apply(lambda m: setattr(m, "task", tt))

                if alphas is not None:
                    model.apply(lambda m: setattr(m, "alphas", alphas))

                output = 0
                for i in range(4):
                    rot_images = torch.rot90(data, i, (2, 3))
                    _, outputs_aux = model(rot_images, joint=True)
                    output += outputs_aux['joint'][:, num_cls * i:num_cls * (i + 1)] / 4.

                if w is not None:
                    output.data = output.data * w[tt] + b[tt]
                
                if temperature is not None:
                    output = F.softmax(output / temperature)

                cil_output_list = torch.cat((cil_output_list, output.data), dim=1)

                scores, _ = output.data.max(1, keepdim=True)
                score_list_task.append(scores)

            score_list_task = torch.cat(score_list_task, dim=1)
            score_list_all.append(score_list_task)

            target = num_cls * task + target
            cil_correct += cil_output_list.argmax(1).eq(target).sum().item()

            output_list.append(cil_output_list.data.detach().cpu())
            label_list.append(target.data.cpu())

            task_pred = cil_output_list.argmax(1) // num_cls
            task_labels = torch.zeros_like(target).to(args.device) + task
            correct_task_pred += task_pred.eq(task_labels).sum().item()

            total += len(target)

        score_list_all = torch.cat(score_list_all)
        output_list = torch.cat(output_list).numpy()
        label_list = torch.cat(label_list).numpy()

        acc = cil_correct / total * 100
        task_acc = correct_task_pred / total * 100

        model.apply(lambda m: setattr(m, "alphas", None))
        model.train()
        return acc, task_acc, score_list_all, output_list, label_list

def auc_csi(model, id_loader, ood_loaders, num_tasks_learned, task, w=None, b=None, temperature=None):
    """
        This function obtain scores for IND and scores for all tasks != IND (i.e. OOD tasks)
        id_loader: is a dataloader of IND
        ood_loaders: is a list of dataloaders of OOD
    """
    if args.cal_pretrain is not None:
        cal_pretrain = torch.load(args.cal_pretrain)
        w = cal_pretrain['w'].to(args.device)
        b = cal_pretrain['b'].to(args.device)
        args.logger.print(w)
        args.logger.print(b)

    num_cls = args.output_size
    model.eval()
    def score_func(x, temperature):
        total_outputs = 0
        for i in range(4):
            x_rot = torch.rot90(x, i, (2, 3))
            _, outputs_aux = model(x_rot, penultimate=True, joint=True)
            total_outputs += outputs_aux['joint'][:, num_cls * i:num_cls * (i + 1)] / 4

        if w is not None:
            total_outputs = total_outputs * w[task] + b[task]

        if temperature is None: temperature = 1

        scores = F.softmax(total_outputs / temperature, dim=1).max(dim=1)[0]
        return scores

    def get_scores(loader, temperature):
        scores = []
        for x, y in loader:
            s = score_func(x, temperature)
            scores.append(s.detach().cpu().numpy())
        return np.concatenate(scores)

    def get_auroc(scores_id, scores_ood):
        scores = np.concatenate([scores_id, scores_ood])
        labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
        return roc_auc_score(labels, scores)

    alphas = (
        torch.zeros(
            [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=False
        )
    )
    alphas[task] = 1
    alphas.requires_grad = True

    model.apply(lambda m: setattr(m, "alphas", alphas))
    model.apply(lambda m: setattr(m, "task", task))

    if alphas is not None:
        model.apply(lambda m: setattr(m, "alphas", alphas))

    # Obtain scores for IND
    auroc_dict = {}
    scores_id = get_scores(id_loader, temperature)

    # Obtain scores for OOD
    for ood, ood_loader in ood_loaders.items():
        scores_ood = get_scores(ood_loader, temperature)
        auroc_dict[ood] = get_auroc(scores_id, scores_ood)

    model.apply(lambda m: setattr(m, "alphas", None))
    model.train()
    return auroc_dict

# gt means ground truth task -- corresponds to GG
def gt(
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
    cil=False,
    temperature=None,
    noise=None,
    til=True,
):
    """
        The following evluations are based on one specific task data.
        task: is the data id of the loader
        noise: only provided for CIL functoins. For AUC functions, noise is not implemented.
                For cil_csi, noise has no effect as it's not implemneted.
    """
    model.zero_grad()
    model.train()


    if cil:
        if args.ood_method is None:
            cil_acc, task_acc, score_list_all, output_list, label_list = adapt_test_cil(model, test_loader, num_tasks_learned, task, temperature, noise)
        elif args.ood_method == 'csi':
            cil_acc, task_acc, score_list_all, output_list, label_list = adapt_test_cil_csi(model, test_loader, num_tasks_learned, task, temperature, noise)
            # score_list_all = None
        else:
            raise NotImplementedError()

    alphas = (
        torch.zeros(
            [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=False
        )
    )
    alphas[task] = 1
    alphas.requires_grad = True

    model.apply(lambda m: setattr(m, "alphas", alphas))
    model.apply(lambda m: setattr(m, "task", task))

    # TIL
    if til:
        if args.ood_method is None:
            test_acc = adapt_test(
                model,
                test_loader,
                alphas,
            )
        elif args.ood_method == 'csi':
            test_acc = adapt_test_csi(
                    model,
                    test_loader,
                    alphas,
                )
        else:
            raise NotImplementedError()
    else:
        test_acc = -999


    model.apply(lambda m: setattr(m, "alphas", None))

    if cil:
        return test_acc, cil_acc, task_acc, score_list_all.cpu().numpy(), [output_list, label_list]
    else:
        return test_acc


# The oneshot minimization algorithm.
def se_oneshot_minimization(
    adaptation_criterion,
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()
    # stopping time tracks how many epochs were required to adapt.
    correct = 0
    task_correct = 0

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)
        denominator = model.num_tasks_learned if args.trainer and "nns" in args.trainer else num_tasks_learned
        # alphas_i contains the "beleif" that the task is i
        alphas = (
                torch.ones(
                    [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True
                )
                / denominator
        )

        model.apply(lambda m: setattr(m, "alphas", alphas))

        # Compute the output
        output = model(data)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)

        # Take the gradient w.r.t objective
        grad = torch.autograd.grad(adaptation_criterion(output, model), alphas)
        value, ind = grad[0].min(dim=0)
        alphas = torch.zeros([args.num_tasks, 1, 1, 1, 1], device=args.device)
        alphas[ind] = 1

        args.logger.print(ind)
        predicted_task = ind.item()
        if predicted_task == task:
            task_correct += 1
        else:
            if args.unshared_labels:
                continue

        # Now do regular testing with inferred task.
        model.apply(lambda m: setattr(m, "alphas", alphas))

        with torch.no_grad():

            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = float(correct) / len(test_loader.dataset)

    args.logger.print(
        f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
    )

    model.apply(lambda m: setattr(m, "alphas", None))

    return test_acc


# The binary minimization algorithm.
def se_binary_minimization(
    adaptation_criterion,
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()
    correct = 0
    task_correct = 0

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)

        # alphas_i contains the "beleif" that the task is i
        alphas = (
                torch.ones(
                    [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True
                )
                / num_tasks_learned
        )
        # store the "good" indecies, i.e. still valid for optimization
        good_inds = torch.arange(args.num_tasks) < num_tasks_learned
        good_inds = good_inds.view(args.num_tasks, 1, 1, 1, 1).to(args.device)
        done = False

        prevent_inf_loop_iter = 0
        while not done:
            prevent_inf_loop_iter += 1
            if prevent_inf_loop_iter > np.log2(args.num_tasks) + 1:
                args.logger.print('InfLoop')
                break
            model.zero_grad()

            model.apply(lambda m: setattr(m, "alphas", alphas))

            # Compute the output.
            output = model(data)

            # Take the gradient w.r.t objective
            grad = torch.autograd.grad(adaptation_criterion(output, model), alphas)

            new_alphas = torch.zeros([args.num_tasks, 1, 1, 1, 1], device=args.device)

            inds = grad[0] <= kth_elt(grad[0][good_inds], args.log_base)
            good_inds = inds * good_inds
            new_alphas[good_inds] = 1.0 / good_inds.float().sum().item()
            alphas = new_alphas.clone().detach().requires_grad_(True)
            if good_inds.float().sum() == 1.0:
                predicted_task = good_inds.flatten().nonzero()[0].item()
                done = True

        if predicted_task == task:
            task_correct += 1
        else:
            if args.unshared_labels:
                continue


        model.apply(lambda m: setattr(m, "alphas", alphas))


        with torch.no_grad():

            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_acc = float(correct) / len(test_loader.dataset)
    task_correct = float(task_correct) / len(test_loader.dataset)

    args.logger.print(
        f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
    )

    model.apply(lambda m: setattr(m, "alphas", None))

    return test_acc

# ABatchE using entropy objective.
def se_be_adapt(
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()
    test_loss = 0
    correct = 0
    data_to_repeat = args.data_to_repeat

    task_correct = 0

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)
        model.apply(lambda m: setattr(m, "task", -1))

        if data.shape[0] >= data_to_repeat:
            rep_data = torch.cat(
                tuple([
                    data[j].unsqueeze(0).repeat(model.num_tasks_learned, 1, 1, 1)
                    for j in range(data_to_repeat)
                ]),
                dim=0
            )

            logits = model(rep_data)

            ent = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1)
            ent_reshape = ent.view(data_to_repeat, num_tasks_learned)
            ent_reshape_mean = ent_reshape.mean(dim=0)
            v, i = ent_reshape_mean.min(dim=0)

            ind = i.item()
            args.logger.print(ind)
        predicted_task = ind
        if predicted_task == task:
            task_correct += 1
        else:
            if args.unshared_labels:
                continue
        model.apply(lambda m: setattr(m, "task", ind))


        with torch.no_grad():

            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_acc = float(correct) / len(test_loader.dataset)
    task_correct = float(task_correct) / len(test_loader.dataset)

    args.logger.print(
        f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
    )

    return test_acc

# ABatchE using M objective.
def se_be_max_adapt(
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()
    correct = 0

    data_to_repeat = args.data_to_repeat

    task_correct = 0

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)
        model.apply(lambda m: setattr(m, "task", -1))
        rep_data = torch.cat(
            tuple([
                data[j].unsqueeze(0).repeat(model.num_tasks_learned, 1, 1, 1)
                for j in range(data_to_repeat)
            ]),
            dim=0
        )

        logits = model(rep_data)
        sm = logits.softmax(dim=1)
        ent, _ = sm.max(dim=1)
        ent_reshape = ent.view(data_to_repeat, num_tasks_learned)
        ent_reshape_mean = ent_reshape.mean(dim=0)

        v, i = ent_reshape_mean.max(dim=0)
        ind = i.item()
        args.logger.print(ind)
        predicted_task = ind
        if predicted_task == task:
            task_correct += 1
        else:
            if args.unshared_labels:
                continue
        model.apply(lambda m: setattr(m, "task", ind))


        with torch.no_grad():

            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = float(correct) / len(test_loader.dataset)

    args.logger.print(
        f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
    )

    return test_acc


def se_oneshot_entropy_minimization(*arg, **kwargs):
    def f(logits, model):
        logits = logits[:args.data_to_repeat]
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1).mean()

    return partial(se_oneshot_minimization, f)(*arg, **kwargs)

def se_oneshot_g_minimization(*arg, **kwargs):
    def f(logits, model):
        logits = logits[:args.data_to_repeat]
        m = (torch.arange(logits.size(1)) < args.real_neurons).float().unsqueeze(0).to(args.device)
        logits = (logits * m).detach() + logits * (1-m)
        return logits.logsumexp(dim=1).mean()

    return partial(se_oneshot_minimization, f)(*arg, **kwargs)

def se_binary_entropy_minimization(*arg, **kwargs):
    def f(logits, model):
        logits = logits[:args.data_to_repeat]
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1).mean()

    return partial(se_binary_minimization, f)(*arg, **kwargs)

def se_binary_g_minimization(*arg, **kwargs):
    def f(logits, model):
        logits = logits[:args.data_to_repeat]
        m = (torch.arange(logits.size(1)) < args.real_neurons).float().unsqueeze(0).to(args.device)
        logits = (logits * m).detach() + logits * (1 - m)
        return logits.logsumexp(dim=1).mean()

    return partial(se_binary_minimization, f)(*arg, **kwargs)

# HopSupSup -- Hopfield recovery.
def hopfield_recovery(
    model, writer, test_loader, num_tasks_learned, task,
):
    model.zero_grad()
    model.train()
    # stopping time tracks how many epochs were required to adapt.
    stopping_time = 0
    correct = 0
    taskname = f"{args.set}_{task}"


    params = []
    for n, m in model.named_modules():
        if isinstance(m, FastHopMaskBN):
            out = torch.stack(
                [
                    2 * module_util.get_subnet_fast(m.scores[j]) - 1
                    for j in range(m.num_tasks_learned)
                ]
            )

            m.score = torch.nn.Parameter(out.mean(dim=0))
            params.append(m.score)

    optimizer = optim.SGD(
        params, lr=500, momentum=args.momentum, weight_decay=args.wd,
    )

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)
        hop_loss = None

        for n, m in model.named_modules():
            if isinstance(m, FastHopMaskBN):
                s = 2 * module_util.GetSubnetFast.apply(m.score) - 1
                target = 2 * module_util.get_subnet_fast(m.scores[task]) - 1
                distance = (s != target).sum().item()
                writer.add_scalar(
                    f"adapt_{taskname}/distance_{n}",
                    distance,
                    batch_idx + 1,
                )
        optimizer.zero_grad()
        model.zero_grad()
        output = model(data)
        logit_entropy = (
            -(output.softmax(dim=1) * output.log_softmax(dim=1)).sum(1).mean()
        )
        for n, m in model.named_modules():
            if isinstance(m, FastHopMaskBN):
                s = 2 * module_util.GetSubnetFast.apply(m.score) - 1
                if hop_loss is None:
                    hop_loss = (
                        -0.5 * s.unsqueeze(0).mm(m.W.mm(s.unsqueeze(1))).squeeze()
                    )
                else:
                    hop_loss += (
                        -0.5 * s.unsqueeze(0).mm(m.W.mm(s.unsqueeze(1))).squeeze()
                    )

        hop_lr = args.gamma * (
            float(batch_idx + 1) / len(test_loader)
        )
        hop_loss =  hop_lr * hop_loss
        ent_lr = 1 - (float(batch_idx + 1) / len(test_loader))
        logit_entropy = logit_entropy * ent_lr
        (logit_entropy + hop_loss).backward()
        optimizer.step()

        writer.add_scalar(
            f"adapt_{taskname}/{num_tasks_learned}/entropy",
            logit_entropy.item(),
            batch_idx + 1,
        )

        writer.add_scalar(
            f"adapt_{taskname}/{num_tasks_learned}/hop_loss",
            hop_loss.item(),
            batch_idx + 1,
        )

    test_acc = adapt_test(
        model,
        test_loader,
        alphas=None,
    )

    model.apply(lambda m: setattr(m, "alphas", None))
    return test_acc
