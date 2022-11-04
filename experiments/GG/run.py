import types
from copy import deepcopy
from multiprocessing import Process, Queue
from itertools import product
import sys, os
import numpy as np
import time
import argparse

sys.path.append(os.path.abspath("."))


def kwargs_to_cmd(kwargs):
    cmd = "python main.py "
    # print('rqwlkejqlkjeq', sys.argv)
    argv = [arg.split('--')[1] for arg in sys.argv if '--' in arg]
    for flag, val in kwargs.items():
        # print(flag, val, type(val))
        if val is None:
            pass
        elif isinstance(val, bool):
            if flag in argv:
                cmd += f"--{flag} "
        else:
            cmd += f"--{flag}={val} "

    return cmd


def run_exp(gpu_num, in_queue):
    while not in_queue.empty():
        try:
            experiment = in_queue.get(timeout=3)
        except:
            return

        before = time.time()

        experiment["multigpu"] = gpu_num
        print(f"==> Starting experiment {kwargs_to_cmd(experiment)}")
        os.system(kwargs_to_cmd(experiment))

        with open("output.txt", "a+") as f:
            f.write(
                f"Finished experiment {experiment} in {str((time.time() - before) / 60.0)}."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-sets', default=0, type=lambda x: [a for a in x.split("|") if a])
    parser.add_argument('--seeds', default=1, type=int)
    parser.add_argument('--data', default='~/data', type=str, help='data directory')
    parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist', 'cifar10', 'cifar100', 'timgnet'])
    parser.add_argument('--config', default='cifar10', type=str)
    parser.add_argument('--name', default=None, type=str) # e.g. --name cifar10_supsup, then files are saved under 'args.log_dir/cifar10_supsup'
    parser.add_argument('--round', default=0, type=int)
    parser.add_argument('--log_dir', default='runs/rn18-supsup', type=str)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--load_path', type=str, default=None, help='model dir to load')
    parser.add_argument('--resume_task', type=int, default=None)
    parser.add_argument('--print_filename', type=str, default='result.txt')
    # parser.add_argument('--amp', default=None, type=str, help="either True or None. For some reason, can't provide boolean to kwargs")
    parser.add_argument("--validation", action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--ood_method', default=None, type=str, choices=['csi', ])
    parser.add_argument("--calibration_task", type=int, default=None)
    parser.add_argument("--cal_pretrain", type=str, default=None, help='path and file name of pre-trained calibration paramters')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument("--softmax", action='store_true')
    parser.add_argument("--odin_T", type=float, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--joint_epochs", type=int, default=100)
    parser.add_argument("--original", action='store_true')
    parser.add_argument("--msp", action='store_true')
    parser.add_argument("--odin", action='store_true')
    parser.add_argument("--cal_batch_size", type=int, default=15) # accidently put 15, was planning to use 16
    parser.add_argument("--cal_epochs", type=int, default=160)
    parser.add_argument("--cal_lr", type=float, default=0.01)
    # parser.add_argument('--no_save', action='store_true')
    # parser.add_argument('--cil', action='store_true')
    args = parser.parse_args()

    gpus = args.gpu_sets
    seeds = list(range(args.seeds))
    data = args.data

    args.config = f"experiments/GG/configs/{args.config}.yaml"
    if args.log_dir is None:
        args.log_dir = 'runs/rn18-supsup'
    experiments = []
    # sparsities = [1, 2, 4, 8, 16, 32] # Higher sparsity values mean more dense subnetworks
    sparsities = [32] # Higher sparsity values mean more dense subnetworks

    # if args.no_save:
    #     args.save
    for sparsity, seed in product(sparsities, seeds):
        if args.name is None:
            args.name = f"id=supsup~seed={seed}~sparsity={sparsity}"
        else:
            args.name = args.name + f'_round_{args.round}'
        kwargs = {
            "config": args.config,
            "name": args.name,
            "sparsity": sparsity,
            "seed": seed,
            "log-dir": args.log_dir,
            "epochs": args.epochs,
            "data": data,
            "dataset": args.dataset,
            "load_path": args.load_path,
            "resume_task": args.resume_task,
            "print_filename": args.print_filename,
            "validation": args.validation,
            "amp": args.amp,
            "ood_method": args.ood_method,
            "calibration_task": args.calibration_task,
            "cal_pretrain": args.cal_pretrain,
            "save": args.save,
            "softmax": args.softmax,
            "odin_T": args.odin_T,
            "noise": args.noise,
            "joint_epochs": args.joint_epochs,
            "original": args.original,
            "msp": args.msp,
            "odin": args.odin,
            "cal_batch_size": args.cal_batch_size,
        }

        experiments.append(kwargs)

    print(experiments)
    # input("Press any key to continue...")
    queue = Queue()

    for e in experiments:
        queue.put(e)

    processes = []
    for gpu in gpus:
        p = Process(target=run_exp, args=(gpu, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
