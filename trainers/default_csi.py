from args import args

import torch
import torch.nn as nn

import utils
import models.transform_layers as TL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def init(args):
    pass

def train(model, writer, train_loader, optimizer, criterion, epoch, task_idx, linear, linear_optim, simclr_aug, data_loader=None):
    model.zero_grad()
    model.train()
    num_cls = args.output_size

    loss_sim_cum, loss_linear_cum = 0, 0

    enabled = False
    if args.amp:
        enabled = True
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler2 = torch.cuda.amp.GradScaler(enabled=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.iter_lim < 0 or len(train_loader) * (epoch - 1) + batch_idx < args.iter_lim:
            batch_size = data.size(0)
            data, target = data.to(device), target.to(device)
            images1, images2 = hflip(data.repeat(2, 1, 1, 1)).chunk(2)

            images1 = torch.cat([torch.rot90(images1, rot, (2, 3)) for rot in range(4)])
            images2 = torch.cat([torch.rot90(images2, rot, (2, 3)) for rot in range(4)])
            images_pair = torch.cat([images1, images2], dim=0)  # 8B

            rot_sim_labels = torch.cat([target + num_cls * i for i in range(4)], dim=0)
            rot_sim_labels = rot_sim_labels.to(device)

            images_pair = simclr_aug(images_pair)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=enabled):
                _, outputs_aux = model(images_pair, simclr=True, penultimate=True)

                simclr = utils.normalize(outputs_aux['simclr'])
                sim_matrix = utils.get_similarity_matrix(simclr)
                loss_sim = utils.Supervised_NT_xent(sim_matrix, labels=rot_sim_labels, temperature=0.07)

                loss_sim_cum += loss_sim.item()

            loss = loss_sim

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # scheduler.step(epoch - 1 + batch_idx / len(train_loader))
            # lr = optimizer.param_groups[0]['lr']

            if batch_idx % args.log_interval == 0:
                num_samples = batch_idx * len(data)
                num_epochs = len(train_loader.dataset)
                percent_complete = 100.0 * batch_idx / len(train_loader)
                args.logger.print(
                    f"Train Epoch: {epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                    f"Loss: {loss.item():.6f}"
                )

                t = (len(train_loader) * epoch + batch_idx) * args.batch_size
                writer.add_scalar(f"train/task_{task_idx}/loss", loss.item(), t)

            ### Post-processing stuffs ###
            with torch.cuda.amp.autocast(enabled=enabled):
                penul_1 = outputs_aux['penultimate'][:batch_size]
                penul_2 = outputs_aux['penultimate'][4 * batch_size:5 * batch_size] # THIS IS ANOTHER ORIGINAL 0 DEGREE IMAGES.
                outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])  # only use original rotation

                outputs_linear_eval = linear(outputs_aux['penultimate'].detach().view(batch_size * 2, -1)) # WHAT'S THIS LINEAR EVALUATION? ANS. IN LABELED MULTI-CLASS, THIS HELPS CONTRASTIVE REPRESENTATION. IS IT NECESSARY FOR TESTNG?
                loss_linear = criterion(outputs_linear_eval, target.repeat(2)) # THUS, LINEAR LAYER IS LEARNING THE H_FLIPPED ORIGINAL IMAGES WITH ITS LABELS. I.E. LEARN THE PAIR (HFLIP(x), x_LABEL) FOR x IN THE BATCH.REPEAT(2)

            if args.amp:
                scaler2.scale(loss_linear).backward()
                scaler2.step(linear_optim)
                scaler2.update()
            else:
                linear_optim.zero_grad()
                loss_linear.backward()
                linear_optim.step()

            loss_linear_cum += loss_linear.item()

    args.logger.print("[LossC %f] [LossSim %f]" % (loss_linear_cum / len(train_loader),
                                        loss_sim_cum / len(train_loader)))

def test(model, writer, criterion, test_loader, epoch, task_idx, marginal=False):
    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    logit_entropy = 0.0
    num_cls = args.output_size

    with torch.no_grad():

        for data, target in test_loader:
            if type(data) == list:
                data = data[0]
            data, target = data.to(device), target.to(device)

            if marginal:
                output = 0
                for i in range(4):
                    rot_data = torch.rot90(data, i, (2, 3))
                    _, outputs_aux = model(rot_data, joint=True)
                    output += outputs_aux['joint'][:, num_cls * i:num_cls * (i + 1)] / 4.
            else:
                output = model(data)

            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            logit_entropy += (
                -(output.softmax(dim=1) * output.log_softmax(dim=1))
                .sum(1)
                .mean()
                .item()
            )
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    logit_entropy /= len(test_loader)
    test_acc = float(correct) / len(test_loader.dataset)

    args.logger.print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n")

    writer.add_scalar(f"test/task_{task_idx}/loss", test_loss, epoch)
    writer.add_scalar(f"test/task_{task_idx}/acc", test_acc, epoch)
    writer.add_scalar(f"test/task_{task_idx}/entropy", logit_entropy, epoch)

    return test_acc

def train_joint(model, joint_linear, criterion, optimizer, scheduler, loader, simclr_aug):
    model.eval()
    loss_joint_cum = 0
    num_cls = args.output_size
    for n, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        images = hflip(images)

        images = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])
        joint_labels = torch.cat([labels + num_cls * i for i in range(4)], dim=0)

        images = simclr_aug(images)
        with torch.no_grad():
            _, outputs_aux = model(images, penultimate=True)
        penultimate = outputs_aux['penultimate'].detach()
        outputs_joint = joint_linear(penultimate).view(images.size(0), -1)
        loss_joint = criterion(outputs_joint, joint_labels)

        optimizer.zero_grad()
        loss_joint.backward()
        optimizer.step()

        loss_joint_cum += loss_joint.item()

    args.logger.print('[lossJ %f]' % (loss_joint_cum / len(loader)))




