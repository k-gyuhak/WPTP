import os, sys
import pathlib
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from args import args
import adaptors
import data
import schedulers
import trainers
import utils

import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import models.transform_layers as TL

from datetime import datetime
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_simclr_augmentation():

    # parameter for resizecrop
    if args.dataset == 'mnist':
        sizes = (28, 28, 1)
    else:
        sizes = (32, 32, 3)

    resize_scale = (0.08, 1.0) # resize scaling factor

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=sizes)

    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop,
    )

    return transform

def main():
    if 'csi' in args.config:
        assert args.ood_method == 'csi'
    else:
        assert args.ood_method != 'csi'

    if args.seed is not None:
        pass

    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}")
    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))
    args.run_base_dir = run_base_dir

    args.n_tasks = args.num_tasks

    args.logger = utils.Logger(args, os.path.splitext(run_base_dir)[0])
    args.logger.now()

    args.logger.print(f"=> Saving data in {run_base_dir}")

    # Get dataloader.

    args.logger.print('\n\n',
                        os.uname()[1] + ':' + os.getcwd(),
                        'python', ' '.join(sys.argv),
                      '\n\n')

    args.logger.print(args)

    data_loader = getattr(data, args.set)()

    cil_tracker = utils.Tracker(args)
    if os.path.exists(f'{run_base_dir}/cil_tracker'):
        cil_tracker.mat = torch.load(f'{run_base_dir}/cil_tracker')

    tp_tracker = utils.Tracker(args)
    if os.path.exists(f'{run_base_dir}/tp_tracker'):
        tp_tracker.mat = torch.load(f'{run_base_dir}/tp_tracker')

    til_tracker = utils.Tracker(args)
    if os.path.exists(f'{run_base_dir}/til_tracker'):
        til_tracker.mat = torch.load(f'{run_base_dir}/til_tracker')

    cal_cil_tracker = utils.Tracker(args)
    if os.path.exists(f'{run_base_dir}/cal_cil_tracker'):
        cal_cil_tracker.mat = torch.load(f'{run_base_dir}/cal_cil_tracker')

    cal_auc_softmax_tracker = [utils.AUCILTracker(args) for _ in range(args.num_tasks)]
    auc_softmax_tracker = [utils.AUCILTracker(args) for _ in range(args.num_tasks)]

    cal_tp_tracker = utils.Tracker(args)

    # Track accuracy on all tasks.
    if args.num_tasks:
        best_acc1 = [0.0 for _ in range(args.num_tasks)]
        curr_acc1 = [0.0 for _ in range(args.num_tasks)]
        adapt_acc1 = [0.0 for _ in range(args.num_tasks)]
        cil_acc1 = [0.0 for _ in range(args.num_tasks)]
        curr_acc1_joint = [0.0 for _ in range(args.num_tasks)]
        task_acc1 = [0.0 for _ in range(args.num_tasks)]
        avg_auc1 = [0.0 for _ in range(args.num_tasks)]

    # Get the model.
    model = utils.get_model()

    # If necessary, set the sparsity of the model of the model using the ER sparsity budget (see paper).
    if args.er_sparsity:
        for n, m in model.named_modules():
            if hasattr(m, "sparsity"):
                m.sparsity = min(
                    0.5,
                    args.sparsity
                    * (m.weight.size(0) + m.weight.size(1))
                    / (
                        m.weight.size(0)
                        * m.weight.size(1)
                        * m.weight.size(2)
                        * m.weight.size(3)
                    ),
                )
                args.logger.print(f"Set sparsity of {n} to {m.sparsity}")

    # Put the model on the GPU,
    model = utils.set_gpu(model)

    criterion = nn.CrossEntropyLoss().to(args.device)

    writer = SummaryWriter(log_dir=run_base_dir)

    # Track the number of tasks learned.
    num_tasks_learned = 0

    if args.ood_method is None:
        trainer = getattr(trainers, args.trainer or "default")
    elif args.ood_method == 'csi':
        trainer = getattr(trainers, "default_csi")
    else:
        raise NotImplementedError()

    args.logger.print(f"=> Using trainer {trainer}") # FOR SPLITCIFAR100, DEFAULT.PY IS USED

    train, test = trainer.train, trainer.test

    # Initialize model specific context (editorial note: avoids polluting main file)
    if hasattr(trainer, "init"): # I THINK, FOR DEFAULT.PY, NOTHING HAPPENS FOR 'INIT'
        trainer.init(args)

    # Iterate through all tasks.
    if args.ood_method == 'csi':
        simclr_aug = get_simclr_augmentation().to(device)

    if args.resume_task is not None:
        assert args.load_path is not None

    for idx in range(args.num_tasks or 0):
        # Optionally resume from a checkpoint.
        # if args.resume:
        if args.load_path:
            load_task = int(args.load_path.split('.pt')[0].split('_')[-1])
            if idx <= load_task:
                if args.calibration_task is not None and idx < args.calibration_task:
                    pass
                elif idx < load_task:
                    pass
                else:
                    resume_path = '/'.join(args.load_path.split('/')[:-1])
                    model_type = args.load_path.split('/')[-1]
                    assert '.pt' in model_type

                    if 'joint' in model_type:
                        resume_path = os.path.join(resume_path, f'result_joint_{idx}.pt')
                    else:
                        resume_path = os.path.join(resume_path, f'result_{idx}.pt')
                    args.logger.print(f"=> Loading checkpoint '{resume_path}'")
                    checkpoint = checkpoint = torch.load(resume_path)
                    best_acc1 = checkpoint["best_acc1"]
                    pretrained_dict = checkpoint["state_dict"]
                    model_dict = model.state_dict()
                    pretrained_dict = {
                        k: v for k, v in pretrained_dict.items() if k in model_dict
                    }
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(pretrained_dict)

        now = datetime.now()
        args.logger.print(now.strftime("%d/%m/%Y %H:%M:%S"), end=' | ')
        args.logger.print(f"Task {args.set}: {idx}")
        total_num_p = 0
        for n, p in model.named_parameters():
            if 'scores' not in n:
                args.logger.print(n, p.numel(), p.requires_grad)
                total_num_p += p.numel()
        args.logger.print("total num param:", total_num_p)

        # Tell the model which task it is trying to solve -- in Scenario NNs this is ignored.
        model.apply(lambda m: setattr(m, "task", idx))

        # Update the data loader so that it returns the data for the correct task, also done by passing the task index.
        assert hasattr(
            data_loader, "update_task"
        ), "[ERROR] Need to implement update task method for use with multitask experiments"

        data_loader.update_task(idx) # THIS UPDATES DATASETS SELF.TRAIN AND SELF.VAL TO CONTAIN THE NEXT TASK DATA

        # Clear the grad on all the parameters.
        for p in model.parameters():
            p.grad = None

        # Make a list of the parameters relavent to this task.
        params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            split = n.split(".")
            if split[-2] in ["scores", "s", "t"] and (
                int(split[-1]) == idx or (args.trainer and "nns" in args.trainer)
            ):
                params.append(p)
            # train all weights if train_weight_tasks is -1, or num_tasks_learned < train_weight_tasks
            if (
                args.train_weight_tasks < 0
                or num_tasks_learned < args.train_weight_tasks
            ):
                if split[-1] == "weight" or split[-1] == "bias":
                    params.append(p)

        # train_weight_tasks specifies the number of tasks that the weights are trained for.
        # e.g. in SupSup, train_weight_tasks = 0. in BatchE, train_weight_tasks = 1.
        # If training weights, use train_weight_lr. Else use lr.
        lr = (
            args.train_weight_lr
            if args.train_weight_tasks < 0
            or num_tasks_learned < args.train_weight_tasks
            else args.lr
        )

        # get optimizer, scheduler
        if args.optimizer == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=args.wd)
        elif args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(params, lr=lr)
        elif args.optimizer == 'lars':
            # from torchlars import LARS
            from lars_optimizer import LARC
            args.logger.print("optimizer == lars")
            base_optimizer = optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.wd)
            optimizer = LARC(base_optimizer, trust_coefficient=0.001)
        else:
            optimizer = optim.SGD(
                params, lr=lr, momentum=args.momentum, weight_decay=args.wd
            )
        
        train_epochs = args.epochs

        if args.no_scheduler:
            scheduler = None
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=train_epochs)
            scheduler_warmup = None
            if args.ood_method == 'csi':
                from trainers.scheduler import GradualWarmupScheduler
                scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=10, after_scheduler=scheduler)

        if args.ood_method == 'csi':
            linear = model.module.linear
            linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999))

        # Train on the current task.
        if (args.load_path is not None and args.resume_task is None) \
            or (args.load_path is not None and idx < args.resume_task):
            args.logger.print("Skip task {}".format(idx))
            epoch = train_epochs
            pass
        else:
            # (args.resume is not None or idx >= args.resume_task)
            # or (args.resume is None or idx < args.resume_task)
            # or (args.resume is None or idx >= args.resume_task)
            args.logger.print("Train task {}".format(idx))
            for epoch in range(1, train_epochs + 1):
                model.train()
                if args.ood_method is None:
                    train(
                        model,
                        writer,
                        data_loader.train_loader,
                        optimizer,
                        criterion,
                        epoch,
                        idx,
                        data_loader,
                    )
                elif args.ood_method == 'csi':
                    train(
                        model,
                        writer,
                        data_loader.train_loader,
                        optimizer,
                        criterion,
                        epoch,
                        idx,
                        linear,
                        linear_optim,
                        simclr_aug,
                        data_loader,
                    )
                else:
                    raise NotImplementedError()

                # Required for our PSP implementation, not used otherwise.
                utils.cache_weights(model, num_tasks_learned + 1)

                curr_acc1[idx] = test(
                    model, writer, criterion, data_loader.val_loader, epoch, idx
                )
                if curr_acc1[idx] > best_acc1[idx]:
                    best_acc1[idx] = curr_acc1[idx]
                if scheduler:
                    if scheduler_warmup:
                        scheduler_warmup.step()
                    else:
                        scheduler.step()

                if (
                    args.iter_lim > 0
                    and len(data_loader.train_loader) * epoch > args.iter_lim
                ):
                    break

        if args.save:
            torch.save(
                {
                    "epoch": args.epochs,
                    "arch": args.model,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "curr_acc1": curr_acc1,
                    "args": args,
                },
                run_base_dir / "result_{}.pt".format(idx),
            )

        utils.write_result_to_csv(
            name=f"{args.name}~set={args.set}~task={idx}",
            curr_acc1=curr_acc1[idx],
            best_acc1=best_acc1[idx],
            save_dir=run_base_dir,
        )

        # Save memory by deleting the optimizer and scheduler.
        if args.ood_method is None:
            del optimizer, scheduler, params
        elif args.ood_method == 'csi':
            del optimizer, scheduler, params, scheduler_warmup
        else:
            raise NotImplementedError()

        # Joint classifier training
        if args.ood_method == 'csi':
            train_joint = trainer.train_joint

            # Train joint linear
            joint_linear = model.module.joint_distribution_layer
            milestones = [int(0.6 * 100), int(0.75 * 100), int(0.9 * 100)]
            joint_linear_optim = torch.optim.SGD(joint_linear.parameters(),
                                                 lr=1e-1, weight_decay=args.wd)
            joint_scheduler = MultiStepLR(joint_linear_optim, gamma=0.1, milestones=milestones)
            if (args.load_path is not None and args.resume_task is None) \
                or (args.load_path is not None and idx < args.resume_task):
                args.logger.print("Skip joint classifier training for task {}".format(idx))
                epoch = args.joint_epochs
                pass
            else:
                args.logger.print("Train joint classifier for task {}".format(idx))
                for epoch in range(args.joint_epochs):
                    model.train()
                    train_joint(model, joint_linear, criterion, joint_linear_optim,
                                joint_scheduler, data_loader.train_loader, simclr_aug)
                    joint_scheduler.step()

                curr_acc1_joint[idx] = test(
                    model, writer, criterion, data_loader.val_loader, epoch, idx, marginal=True
                )

            if args.save:
                torch.save(
                    {
                        "epoch": args.epochs,
                        "arch": args.model,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "curr_acc1": curr_acc1,
                        "args": args,
                    },
                    run_base_dir / "result_joint_{}.pt".format(idx),
                )

        # Increment the number of tasks learned.
        num_tasks_learned += 1

        # If operating in NNS scenario, get the number of tasks learned count from the model.
        if args.trainer and "nns" in args.trainer:
            model.apply(
                lambda m: setattr(
                    m, "num_tasks_learned", min(model.num_tasks_learned, args.num_tasks)
                )
            )
        else:
            model.apply(lambda m: setattr(m, "num_tasks_learned", num_tasks_learned))

        # Calibration training
        if args.calibration_task is not None:
            if num_tasks_learned != args.calibration_task + 1:
                continue
            else:
                args.logger.print("Calibration training")
                num_cls = args.output_size
                cali_loaders = []
                for j in range(num_tasks_learned):
                    data_loader.update_task(j)
                    cali_loader = data_loader.cal_loader
                    cali_loaders.append(cali_loader)

                counter = 0
                w = torch.rand(num_tasks_learned, requires_grad=False, device=device)# / num_tasks_learned
                b = torch.rand(num_tasks_learned, requires_grad=False, device=device)# / num_tasks_learned
                w.requires_grad = True
                b.requires_grad = True
                lr = args.cal_lr
                optimizer = torch.optim.SGD([w, b], lr=lr, momentum=0.8)
                model.eval()
                for epoch in range(args.cal_epochs * len(cali_loaders[0])):
                    output_list, label_list = [], []
                    for t_loader, loader in enumerate(cali_loaders):
                        images, labels = iter(loader).next()
                        images, labels = images.to(device), labels.to(device)
                        labels = num_cls * t_loader + labels

                        cil_output_list = torch.tensor([]).to(args.device)
                        for task_id in range(num_tasks_learned):
                            alphas = (
                                torch.zeros(
                                    [args.num_tasks, 1, 1, 1, 1], device=device, requires_grad=False
                                )
                            )
                            alphas[task_id] = 1
                            alphas.requires_grad = True

                            model.apply(lambda m: setattr(m, "alphas", alphas))
                            model.apply(lambda m: setattr(m, "task", task_id))

                            output = 0
                            model.eval()
                            for i in range(4):
                                with torch.no_grad():
                                    rot_images = torch.rot90(images, i, (2, 3))
                                    _, outputs_aux = model(rot_images, joint=True)
                                    output += outputs_aux['joint'][:, num_cls * i:num_cls * (i + 1)] / 4.

                            output = output * w[task_id] + b[task_id]
                            cil_output_list = torch.cat((cil_output_list, output), dim=1)
                        output_list.append(cil_output_list)
                        label_list.append(labels)
                    output_list = torch.cat(output_list)
                    label_list = torch.cat(label_list)

                    loss = criterion(output_list, label_list)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    args.logger.print(loss.item())
                    if epoch % 40 == 0:
                        args.logger.print(epoch, loss.item())
                        acc_list, task_acc_list = [], []
                        avg_auc1_cal = [0.0 for _ in range(args.num_tasks)]
                        for t in range(num_tasks_learned):
                            args.logger.print(t)
                            data_loader.update_task(t)
                            val_loader = data_loader.val_loader
                            acc, task_acc, score_list_all, output_list, label_list = adaptors.adapt_test_cil_csi(model, val_loader, num_tasks_learned, t, w, b)
                            acc_list.append(acc)
                            cal_cil_tracker.update(acc, num_tasks_learned - 1, t)
                            task_acc_list.append(task_acc)

                        args.logger.print("Average Acc: {:.4f}".format(np.array(acc_list).mean()))
                        args.logger.print("Average Task Prediction: {:.4f}".format(np.array(task_acc_list).mean()))

                        args.logger.print("Calibration softmax AUC result")
                        cal_auc_softmax_tracker[num_tasks_learned - 1].print_result(num_tasks_learned - 1, type='acc')
                        args.logger.print("Calibration CIL result")
                        cal_cil_tracker.print_result(task_id, type='acc')
                        cal_cil_tracker.print_result(task_id, type='forget')
                        args.logger.print()

                        args.logger.print(w)
                        args.logger.print(b)

                        torch.save({'w': w, 'b': b}, args.logger.dir() + f'/calibration_{num_tasks_learned - 1}')
                        counter += 1
                    if counter == 4:
                        # torch.save(cl_outputs, './' + P.logout + '/cl_outputs_adapt_w_b')
                        sys.exit()
                        # break
                model.train()

        # TODO series of asserts with required arguments (eg num_tasks)
        # args.eval_ckpts contains values of num_tasks_learned for which testing on all tasks so far is performed.
        # this is done by default when all tasks have been learned, but you can do something like
        # args.eval_ckpts = [5,10] to also do this when 5 tasks are learned, and again when 10 tasks are learned.

        args.logger.now()
        # Test all the learned tasks
        outputs_list, labels_list = [], []
        score_dict = {}
        if num_tasks_learned in args.eval_ckpts or num_tasks_learned <= args.num_tasks: # NOTE: args.eval_ckpts=[]
            if args.load_path is not None:
                if args.resume_task is None:
                    if num_tasks_learned < load_task + 1: continue
                    elif num_tasks_learned == load_task + 1: pass
                    elif num_tasks_learned > load_task + 1: sys.exit()
                    else: raise NotImplementedError()
                else:
                    if num_tasks_learned < args.resume_task: continue
            avg_acc = 0.0
            avg_correct = 0.0

            # Settting task to -1 tells the model to infer task identity instead of being given the task.
            model.apply(lambda m: setattr(m, "task", -1))

            # an "adaptor" is used to infer task identity.
            # args.adaptor == gt implies we are in scenario GG.

            # This will cache all of the information the model needs for inferring task identity.
            if args.adaptor != "gt":
                utils.cache_masks(model)

            # Iterate through all tasks.
            adapt = getattr(adaptors, args.adaptor) # ARGS.ADAPTOR IS 'GT'

            # Create auxilary tracker matrices so we don't write if-else everytime
            if args.cal_pretrain is not None:
                some_cil_tracker = deepcopy(cal_cil_tracker)
                some_tp_tracker = deepcopy(cal_tp_tracker)
                some_auc_tracker = deepcopy(cal_auc_softmax_tracker)
            else:
                some_cil_tracker = deepcopy(cil_tracker)
                some_tp_tracker = deepcopy(tp_tracker)
                some_auc_tracker = deepcopy(auc_softmax_tracker)

            for i in range(num_tasks_learned):
                args.logger.print(f"Testing {i}: {args.set} ({i})")

                # Update the data loader so it is returning data for the right task.
                data_loader.update_task(i)

                # Clear the stored information -- memory leak happens if not.
                for p in model.parameters():
                    p.grad = None

                for b in model.buffers():
                    b.grad = None

                torch.cuda.empty_cache()

                # TIL and standard CIL
                adapt_acc, cil_acc, task_acc, score_list_all, output_label = adapt(
                    model,
                    writer,
                    data_loader.val_loader,
                    num_tasks_learned,
                    i,
                    True,
                    temperature=None
                )

                adapt_acc1[i] = adapt_acc * 100
                til_tracker.update(adapt_acc * 100, num_tasks_learned - 1, i)
                cil_acc1[i] = cil_acc
                some_cil_tracker.update(cil_acc, num_tasks_learned - 1, i)
                task_acc1[i] = task_acc
                some_tp_tracker.update(task_acc, num_tasks_learned - 1, i)
                avg_acc += adapt_acc
                score_dict[i] = score_list_all

                outputs_list.append(output_label[0])
                labels_list.append(output_label[1])

                torch.cuda.empty_cache()
                utils.write_adapt_results(
                    name=args.name,
                    task=f"{args.set}_{i}",
                    num_tasks_learned=num_tasks_learned,
                    curr_acc1=curr_acc1[i],
                    adapt_acc1=adapt_acc,
                    task_number=i,
                )

            for task_id in range(num_tasks_learned):
                utils.auc(score_dict, task_id, some_auc_tracker[num_tasks_learned - 1])

            # AUC
            args.logger.print("Softmax AUC result")
            # auc_softmax_tracker[num_tasks_learned - 1].print_result(num_tasks_learned - 1, type='acc')
            some_auc_tracker[num_tasks_learned - 1].print_result(num_tasks_learned - 1, type='acc')

            # TIL
            args.logger.print("TIL result")
            til_tracker.print_result(num_tasks_learned - 1, type='acc')
            til_tracker.print_result(num_tasks_learned - 1, type='forget')
            args.logger.print()

            # CIL
            args.logger.print("CIL result")
            # cil_tracker.print_result(num_tasks_learned - 1, type='acc')
            # cil_tracker.print_result(num_tasks_learned - 1, type='forget')
            some_cil_tracker.print_result(num_tasks_learned - 1, type='acc')
            some_cil_tracker.print_result(num_tasks_learned - 1, type='forget')

            # TP
            args.logger.print("TP result")
            # tp_tracker.print_result(num_tasks_learned - 1, type='acc')
            # tp_tracker.print_result(num_tasks_learned - 1, type='forget')
            some_tp_tracker.print_result(num_tasks_learned - 1, type='acc')
            some_tp_tracker.print_result(num_tasks_learned - 1, type='forget')

            if args.cal_pretrain is not None:
                cal_cil_tracker = deepcopy(some_cil_tracker)
                cal_tp_tracker = deepcopy(some_tp_tracker)
                cal_auc_softmax_tracker = deepcopy(some_auc_tracker)
            else:
                cil_tracker = deepcopy(some_cil_tracker)
                tp_tracker = deepcopy(some_tp_tracker)
                auc_softmax_tracker = deepcopy(some_auc_tracker)

            writer.add_scalar(
                "adapt/avg_acc", avg_acc / num_tasks_learned, num_tasks_learned
            )
            writer.add_scalar(
                "avg_cil_acc", np.array(cil_acc1).mean(), num_tasks_learned
            )
            
            utils.clear_masks(model)
            torch.cuda.empty_cache()

        # CIL and TIL
        args.logger.print("Avg TIL acc: {:.4f}".format(np.array(adapt_acc1[:num_tasks_learned]).mean()))
        args.logger.print("Avg CIL acc: {:.4f}".format(np.array(cil_acc1[:num_tasks_learned]).mean()))
        args.logger.print("Avg AUC acc: {:.4f}".format(np.array(avg_auc1[:num_tasks_learned]).mean()))
        args.logger.print("Avg Task prediction acc: {:.4f}".format(np.array(task_acc1[:num_tasks_learned]).mean()))

        torch.save(til_tracker.mat, args.logger.dir() + '/til_tracker')
        torch.save(cal_cil_tracker.mat, args.logger.dir() + '/cal_cil_tracker')
        torch.save(cil_tracker.mat, args.logger.dir() + '/cil_tracker')

        # TP
        torch.save(cal_tp_tracker.mat, args.logger.dir() + '/cal_tp_tracker')
        torch.save(tp_tracker.mat, args.logger.dir() + '/tp_tracker')

        # AUC
        auc_mat_list = [val.mat for val in auc_softmax_tracker]
        torch.save(auc_mat_list, args.logger.dir() + f'/auc_softmax_tracker_list')

        torch.save([outputs_list, labels_list], args.logger.dir() + f'/outputs_labels_list_{num_tasks_learned - 1}')

    if args.save:
        torch.save(
            {
                "epoch": args.epochs,
                "arch": args.model,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "curr_acc1": curr_acc1,
                "args": args,
            },
            run_base_dir / "final.pt",
        )

    args.logger.now()

    return adapt_acc1


# TODO: Remove this with task-eval
def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            args.logger.print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            args.logger.print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": args.wd,},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov=False,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.wd,
        )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    return optimizer


if __name__ == "__main__":
    main()
