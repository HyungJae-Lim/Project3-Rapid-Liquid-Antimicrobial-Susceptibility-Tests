import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from glob import glob
from utils import get_confusion
from .BaseRunner import BaseRunner


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn

    return pp


class SpeckleRunner(BaseRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger):
        super().__init__(arg, torch_device, logger)
        self.net = net
        self.loss = loss
        self.optim = optim
        self.description = arg.description
        self.arg = arg

        self.best_metric = -1
        self.start_time = time.time()

        if arg.resume or arg.test:
            self.load(arg.load_fname)


    def save(self, epoch, filename):
        if epoch < 0:
            return

        torch.save({"model_type": self.model_type,
                    "start_epoch": epoch + 1,
                    "network": self.net.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("[%dE] Saved Best Model, Accuracy=%.05f" % (epoch, self.best_metric))


    def load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[-1])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File" % (self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

            self.net.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f" % (
            ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")


    def train(self, train_loader, val_loader=None, test_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.epoch):
            if epoch != 0 and epoch % 50 == 0 and epoch < 350:
                self.arg.lr /= 2
                for p in self.optim.param_groups:
                    p['lr'] = self.arg.lr
                self.logger.will_write("lr decay : %f" % (self.arg.lr))

            train_acc = []
            print('=' * os.get_terminal_size().columns)
            self.net.train()
            train_loader_iter = iter(train_loader)
            if len(self.description) > 0:
                print('[Ongoing Thread] \"{}\" Task'.format(self.description))

            print('[Scheduler] Learning Rate: {}'.format(self.optim.param_groups[0]['lr']))
            pbar = tqdm(range(len(train_loader.dataset)//self.arg.batch_size), smoothing=0.9)
            for train_b_id in pbar:
                inp, tgt, path = next(train_loader_iter)
                tgt = torch.Tensor(list(map(int, tgt))).type(torch.LongTensor)
                tgt = tgt.to(self.torch_device)
                inp = inp.to(self.torch_device).squeeze(2)
                inp = inp/255.

                prediction, loss, ce, l1 = self.net(inp, tgt)
                decision = prediction.max(dim=-1)[1]

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                correct = torch.sum(tgt == decision).float().cpu().item()
                accuracy = correct / self.arg.batch_size
                train_acc.append(accuracy)

                pbar.set_description_str(
                    desc="[Train] Epoch {}/{}, loss:{:.4f}, CE/L1:{:.4f}/{:.4f}, Accuracy:{:.3f}".format(
                        epoch, self.epoch-self.start_epoch-1, loss.item(), ce.item(), l1.item(), accuracy
                    ), refresh=True
                )

            avg_train_acc = sum(train_acc) / len(train_acc)

            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader, avg_train_acc)
            else:
                self.save(epoch)


    def _get_acc(self, loader, confusion=False, test=False, num_labels=2):
        con_count = 0
        uncon_count = 0

        correct = 0
        con_correct = 0
        uncon_correct = 0

        preds = []
        labels = []
        con_stack = []
        uncon_stack = []

        loader_iter = iter(loader)
        pbar = tqdm(range(len(loader.dataset)//self.arg.batch_size_test), smoothing=0.9)
        decision_collections = torch.zeros((num_labels, num_labels))

        for eval_b_id in pbar:
            inp, tgt, path = next(loader_iter)
            tgt = torch.Tensor(list(map(int, tgt))).type(torch.LongTensor)
            tgt = tgt.to(self.torch_device)
            inp = inp.to(self.torch_device).squeeze(2)
            inp = inp/255.

            prediction, _, _, _ = self.net(inp, tgt)
            decision = prediction.max(dim=-1)[1]

            correct += torch.sum(tgt==decision).float().cpu().item()
            con_tmp = decision[decision==0]
            target_tmp = tgt[decision==0]
            con_correct += torch.sum((con_tmp == target_tmp).int())
            con_count += torch.sum((tgt==0).int())

            uncon_tmp = decision[decision==1]
            target_tmp = tgt[decision==1]
            uncon_correct += torch.sum((uncon_tmp == target_tmp).int())
            uncon_count += torch.sum((tgt==1).int())

            con = con_correct / (con_count + 1e-4)
            uncon = uncon_correct / (uncon_count + 1e-4)
            preds += decision.view(-1).tolist()
            labels += tgt.view(-1).tolist()

            if self.arg.test:
                pbar.set_description_str(desc="[TEST] Accuracy: {:.3f} | Contam: {:.3f} | Uncontam: {:.3f}".format(
                    correct / self.arg.batch_size_test / (eval_b_id+1), con, uncon)
                    , refresh=True
                )

            else:
                pbar.set_description_str(desc="[Valid] Accuracy: {:.3f} | Contam: {:.3f} | Uncontam: {:.3f}".format(
                    correct / self.arg.batch_size_test / (eval_b_id+1), con, uncon)
                    , refresh=True
                )

            if self.arg.test:
                if self.arg.flow:
                    num = 499
                else:
                    num = 500 // self.arg.clip_frames
#                    num = 1

            num=500
            if eval_b_id != 0 and len(preds) % num == 0:
                temp = torch.Tensor(preds[-num:]).cuda()
                target = (tgt[-1]/2 + 0.5).long()
                uniques, counts = torch.unique(temp, sorted=True, return_counts=True)
                uniques = (uniques/2 + 0.5).long()
                decision_idx = torch.max(counts, dim=0)[1].item()
                decision_collections[target][uniques[decision_idx]] += 1


        confusion = get_confusion(preds, labels)
        acc = confusion / (1e-7 + np.sum(confusion, axis=-1))
        acc = acc.diagonal()
        print('\n\n' + '=' * os.get_terminal_size().columns)
        print("\nTotal Vote:\n{}\n".format(confusion))
        print("Total Decisions:\n{}\n".format(decision_collections.detach().cpu().numpy()))
        print("Total Accuracy:\n{}\n".format(np.round(acc, 3) * 100))
        print('=' * os.get_terminal_size().columns + '\n\n')

        return correct / (len(loader.dataset) + 1e-6), confusion


    def valid(self, epoch, val_loader, test_loader, train_acc):
        self.net.eval()
        with torch.no_grad():
            acc, *_ = self._get_acc(val_loader)
            self.logger.will_write("[valid] epoch:{}, acc:{}".format(epoch, acc))
            print('[Metric] Train | Valid Accuracy: {:.2f}% | {:.2f}%'.format(train_acc*100, acc*100))

            if acc > self.best_metric: # or (epoch + 1) % 5 == 0:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]" % (epoch, acc))
