import evaluate
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from d2l import torch as d2l
from absl import logging

accuracy = evaluate.load('accuracy')
recall = evaluate.load('recall')
precision = evaluate.load('precision')
f1 = evaluate.load('f1')


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def evaluate_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device

    # metrics: accuracy, recall, precision, f1-score
    avg_acc, avg_recall, avg_precision, avg_f1 = [], [], [], []
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            elif isinstance(X, dict):
                t = dict()
                for k, v in X.items():
                    try:
                        t[k] = v.to(device)
                    except AttributeError:
                        t[k] = v
                X = t
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = torch.max(F.softmax(net(X)), 1)[1]
            eval_acc = accuracy.compute(references=y, predictions=y_hat)['accuracy']
            eval_recall = recall.compute(references=y, predictions=y_hat,
                                         average='macro')['recall']
            eval_precision = precision.compute(references=y, predictions=y_hat,
                                               average='macro')['precision']
            eval_f1 = f1.compute(references=y, predictions=y_hat,
                                 average='macro')['f1']
            avg_acc.append(eval_acc)
            avg_recall.append(eval_recall)
            avg_precision.append(eval_precision)
            avg_f1.append(eval_f1)
    avg_acc = np.array(avg_acc).mean()
    avg_recall = np.array(avg_recall).mean()
    avg_precision = np.array(avg_precision).mean()
    avg_f1 = np.array(avg_f1).mean()
    return avg_acc, avg_recall, avg_precision, avg_f1


def train_batch(model, X, y, loss, trainer, devices, scaler=None):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    elif isinstance(X, dict):
        for k, v in X.items():
            try:
                X[k] = v.to(devices[0])
            except AttributeError:
                X[k] = v
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    model.train()
    trainer.zero_grad()
    pred = model(X)
    l = loss(pred, y.long())
    l.sum().backward()
    if scaler is None:
        trainer.step()
    else:
        scaler.scale(l.sum()).backward()
        scaler.step(trainer)
        scaler.update()
    train_loss_sum = l.sum()
    y_hat = torch.max(F.softmax(pred), 1)[1]
    train_acc = accuracy.compute(references=y, predictions=y_hat)['accuracy']
    train_recall = recall.compute(references=y, predictions=y_hat,
                                  average='macro')['recall']
    train_precision = precision.compute(references=y, predictions=y_hat,
                                        average='macro')['precision']
    return train_loss_sum, train_acc, train_recall, train_precision


def train(model, train_iter, test_iter,
          loss, trainer,
          num_epochs,
          devices=d2l.try_all_gpus(),
          use_scaler=False):
    timer, num_batches = d2l.Timer(), len(train_iter)
    writer = SummaryWriter('/root/tf-logs')
    animator = d2l.Animator(xlabel='epoch',
                            xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'train recall', 'train precision', 'test acc'])
    model = model.to(devices[0])
    for epoch in range(num_epochs):
        # 6-dim metrics: train loss, train acc, train recall, train precision, #samples, #features
        metric = d2l.Accumulator(6)
        train_losses, train_accs, train_recalls, train_precisions = [], [], [], []
        scaler = torch.cuda.amp.GradScaler() if use_scaler else None
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc, recall, precision = train_batch(
                model, features, labels, loss, trainer, devices, scaler)
            metric.add(l, acc, recall, precision, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[4],
                              metric[1], metric[2], metric[3],
                              None))
                train_losses.append(metric[0] / metric[4])
                train_accs.append(acc)
                train_recalls.append(recall)
                train_precisions.append(precision)
        # metrics on train samples per epoch
        avg_train_loss = np.array(train_losses).mean()
        avg_train_acc = np.array(train_accs).mean()
        avg_train_recall = np.array(train_recalls).mean()
        avg_train_precision = np.array(train_precisions).mean()
        logging.info("Epoch {:d}: Avg train loss {:.6f}, Avg train acc {:.3f}, Avg train recall {:.3f}, Avg train "
                     "precision {:.3f}".format(epoch,
                                               avg_train_loss,
                                               avg_train_acc,
                                               avg_train_recall,
                                               avg_train_precision))

        # metrics on test samples per epoch
        test_acc, test_recall, test_precision, test_f1 = evaluate_gpu(model, test_iter)
        logging.info("Epoch {:d}: "
                     "Val acc {:.3f}, Val recall {:.3f}, Val precision {:.3f}, Val f1 {:.3f}".format(epoch,
                                                                                                     test_acc,
                                                                                                     test_recall,
                                                                                                     test_precision,
                                                                                                     test_f1))
        # d2l animator setting
        animator.add(epoch + 1, (None, None, None, None, test_acc))

        # visualize setting
        writer.add_scalar("Train Loss", avg_train_loss, epoch)
        writer.add_scalars("Train Metrics",
                           {"Avg train acc": avg_train_acc,
                            "Avg train recall": avg_train_recall,
                            "Avg train precision": avg_train_precision},
                           epoch)
        writer.add_scalars("Validation Metrics",
                           {"Val acc": test_acc,
                            "Val recall": test_recall,
                            "Val precision": test_precision,
                            "Val F1 Score": test_f1},
                           epoch)
    animator.fig.show()
    logging.info(f'{metric[4] * num_epochs / timer.sum():.1f} examples/sec on '
                 f'{str(devices)}')


def test(model, test_iter, devices=d2l.try_all_gpus()):
    avg_acc, avg_recall, avg_precision, avg_f1 = evaluate_gpu(model, test_iter, devices[0])
    logging.info(f'Average accuracy: {avg_acc}')
    logging.info(f'Average recall: {avg_recall}')
    logging.info(f'Average precision: {avg_precision}')
    logging.info(f'Average macro f1: {avg_f1}')
    return avg_acc, avg_recall, avg_precision, avg_f1

