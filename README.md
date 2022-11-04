# Overview

This is the official repository of A Theoretical Study on Solving Continual Learning (NeurIPS 2022).

In this repository, we provide the codes for Sup+CSI and Sup+CSI+c. For HAT+CSI and HAT+CSI+c, please check the repository of our workshop paper ([Link](https://github.com/k-gyuhak/CLOM))

# Note
Check out our related paper

- A Multi-Head Model for Continual Learning via Out-of-Distribution Replay, CoLLAs 2022 [[PDF](https://arxiv.org/abs/2208.09734)]

# Environments

The code has been tested on the machine with

RTX 3090
- cuda=11.4
- pytorch=1.7.1
- torchvision=0.8.2
- cudatoolkit=11.0.221
- tensorboardx=2.1
- diffdist=0.1
- gdown=4.4.0

Please install the necessary packages

# Training
Please change --dataset --config --log_dir --name for different experiments
```
python experiments/GG/run.py --data='./data' --dataset 'cifar10' --config 'cifar10_5t_csi' --log_dir './logs/cifar10_5t_csi' --name 'cifar10_5t_csi' --ood_method 'csi' --round 0 --gpu-sets=0 --seeds 1 --save --original --amp
```
choices of dataset: mnist, cifar10, cifar100, timgnet
choices of config: mnist_5t_csi, cifar10_5t_csi, cifar100_10t_csi, cifar100_20t_csi, timgnet_5t_csi, timgnet_10t_csi

Without mixed precision, remove --amp

To resume training,
```
python experiments/GG/run.py --data './data' --dataset 'cifar10' --config 'cifar10_5t_csi' --log_dir './logs/cifar10_5t_csi' --load_path './logs/cifar10_5t_csi/cifar10_5t_csi_round_0/result_joint_3.pt' --name 'cifar10_5t_csi' --ood_method 'csi' --round 0 --gpu-sets=0 --seeds 1 --resume_task 4 --original
```
Please change load_path and resume_task for your experiment.
 

# Training calibration parameters
Please change --dataset, --config --log_dir, --load_path, --name, --calibration_task
```
python experiments/GG/run.py --data './data' --dataset 'cifar10' --config 'cifar10_5t_csi' --log_dir './logs/cifar10_5t_csi' --load_path './logs/cifar10_5t_csi/cifar10_5t_csi_round_0/result_joint_4.pt' --name 'cifar10_5t_csi' --print_filename 'calibration_4.txt' --ood_method 'csi' --round 0 --gpu-sets=0 --seeds 1 --original --calibration_task 4
```
The numbers for --calibration_task and result_joint_NUMBER.pt in --load_path must be the same

# Evaluation without calibration
```
python experiments/GG/run.py --data './data' --dataset 'cifar10' --config 'cifar10_5t_csi' --log_dir './logs/cifar10_5t_csi' --load_path './logs/cifar10_5t_csi/cifar10_5t_csi_round_0/result_joint_4.pt' --name 'cifar10_5t_csi' --print_filename 'eval_without_cal.txt' --round 0 --gpu-sets=0 --seeds 1 --ood_method 'csi' --original
```

# Evaluation using calibration
```
python experiments/GG/run.py --data './data' --dataset 'cifar10' --config 'cifar10_5t_csi' --log_dir './logs/cifar10_5t_csi' --load_path './logs/cifar10_5t_csi/cifar10_5t_csi_round_0/result_joint_4.pt' --name 'cifar10_5t_csi' --cal_pretrain './logs/cifar10_5t_csi/cifar10_5t_csi_round_0/calibration_4' --print_filename 'eval_with_cal.txt' --round 0 --gpu-sets=0 --seeds 1 --ood_method 'csi' --original
```



# Evaluation using pre-trained models
Please download the pre-trained models and calibration parameters by running download_pretrained_models.py or download manually from [link](https://drive.google.com/drive/folders/1V55POcW9JJEW1mMRPq_GeFD1sfDYNaHi). The models and calibration parameters need to be saved under LOG_DIR/NAME_round_NO/, where LOG_DIR is --log_dir (e.g., ./logs/cifar10_5t_csi) and NAME is --name (e.g., cifar10_5t_csi) and NO is the round number (e.g., usually 0).

The provided pre-trained models give the following CIL results

|          | MNIST |  CIFAR10 | CIFAR100-10T | CIFAR100-20t | T-ImageNet-5T | T-ImageNet-10T |
| ---------| ------| ----- | ----- | ----- | ----- | ----- |
| Sup+CSI | 79.62 | 85.52 | 64.63 | 59.28 | 49.25 | 45.81 |
| Sup+CSI+c     | 82.51 | 87.75 | 64.67 | 59.42 | 49.52 | 45.96 |

# Different prediction method (Appendix C)
For the CIL prediction method explained in Appendix C, please run 
```
python cil_is_til_times_tp.py
```

# Acknowledgement
The code uses the source code from [CSI](https://github.com/alinlab/CSI) and [SupSup](https://github.com/RAIVNLab/supsup).
