import os
import sys
import time
import math
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F


class Appr(object):
    def __init__(self, model, data, sbatch=64, clipgrad=100, nepochs=150, lr=1e-3, beta=1, lamb=1, checkpoint_path=None):
        self.model = model
        self.data = data

        self.sbatch = sbatch
        self.clipgrad = clipgrad

        self.nepochs = nepochs
        self.lr = lr
        self.lamb = lamb
        self.beta = beta

        self.checkpoint_path = checkpoint_path
        assert self.checkpoint_path != None

    def train_task(self, t, eval_segment=1):
        self.model.add_task_body_params(t)
        self.optimizer = torch.optim.Adam(self.model.get_task_specific_parameters(t), lr=self.lr)
        self.epoch = -1

        # Check for current running checkpoint
        checkpoint_path = f"{self.checkpoint_path}_task{t+1}.tar"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            print(f"Loaded current checkpoint task {t+1}, epoch {checkpoint['epoch']+1}")

        xtrain = torch.cat([self.data[t]["train"]["x"].cuda(), self.data[t]["valid"]["x"].cuda()], dim=0)
        ytrain = torch.cat([self.data[t]["train"]["y"].cuda(), self.data[t]["valid"]["y"].cuda()], dim=0)

        nsamples_0 = self.data[0]["train"]["x"].shape[0] + self.data[0]["valid"]["x"].shape[0]
        nsamples_t = self.data[t]["train"]["x"].shape[0] + self.data[t]["valid"]["x"].shape[0]

        num_epochs_to_train = math.ceil(self.nepochs * nsamples_0 / nsamples_t)
        eval_segment = math.ceil(eval_segment * nsamples_0 / nsamples_t)
        print(f"***Training task {t+1} ({self.data[t]['name']}) - {num_epochs_to_train} epochs***")

        for self.epoch in range(self.epoch + 1, num_epochs_to_train):
            clock = time.time()
            class_loss, kl_term, total_loss, train_acc = self.train_epoch(t, xtrain, ytrain)
            print(f"| Epoch {self.epoch+1}/{num_epochs_to_train} | Time={time.time()-clock:5.1f}s | class_loss={class_loss:.5f} | kl_term={kl_term:.5f} | total_loss={total_loss:.5f} | acc={100*train_acc:5.1f}% |")

            # Save current running checkpoint
            checkpoint = {"task": t, "epoch": self.epoch, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved current checkpoint {t+1}, epoch {checkpoint['epoch']+1}")

            # Display evaluate
            if (self.epoch + 1) % eval_segment == 0:
                self.test_to_task(t, num_samples=16)

    def test_to_task(self, t, num_samples=32):
        accs = []
        for i in range(t + 1):
            xtest = self.data[i]["test"]["x"].cuda()
            ytest = self.data[i]["test"]["y"].cuda()
            acc = self.test_task(i, xtest, ytest, num_samples=num_samples)
            accs.append(acc)
            print(f">>> Test on task {i+1} - {self.data[i]['name']}: acc={100*acc:5.1f}% <<<")
        return np.array(accs)

    def train_epoch(self, t, x, y, num_samples=16):
        self.model.train()

        r = np.arange(x.shape[0])
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        epoch_class_loss = 0
        epoch_kl_term = 0
        epoch_total_loss = 0
        total_hits = 0

        # Loop batches
        parameters = self.model.get_task_specific_parameters(t)

        for i in range(0, len(r), self.sbatch):
            # Batch
            images = x[r[i : min(len(r), i + self.sbatch)]]
            targets = y[r[i : min(len(r), i + self.sbatch)]]

            # Forward current model
            outputs = self.model(images, int(t) * torch.ones_like(targets), tasks=[t], num_samples=num_samples)
            output = outputs[t]

            # Calculate loss
            flattened_output = output.view(-1, output.shape[-1])
            stacked_targets = targets.repeat([num_samples])
            class_loss = F.cross_entropy(flattened_output, stacked_targets, reduction="mean")

            # Scale kl_term
            kl_term = self.beta * self.model.get_kl_task(t, lamb=self.lamb) / x.shape[0]
            loss = class_loss + kl_term

            # Sum
            epoch_class_loss += class_loss.detach().data.item()
            epoch_kl_term += kl_term.detach().data.item()
            epoch_total_loss += loss.detach().data.item()

            # Accuracy
            probs = F.softmax(output, dim=2).mean(dim=0)
            _, pred = probs.max(1)
            hits = (pred == targets).float()
            total_hits += hits.sum().data.cpu().numpy().item()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, self.clipgrad)
            self.optimizer.step()

        return epoch_class_loss / len(r), epoch_kl_term / len(r), epoch_total_loss / len(r), total_hits / len(r)

    def test_task(self, t, x, y, num_samples=32):
        with torch.no_grad():
            total_hits = 0
            self.model.eval()

            r = np.arange(x.shape[0])
            r = torch.LongTensor(r).cuda()

            # Loop batches
            for i in range(0, len(r), self.sbatch):
                # Batch
                images = x[r[i : min(len(r), i + self.sbatch)]]
                targets = y[r[i : min(len(r), i + self.sbatch)]]

                # Forward
                outputs = self.model(images, int(t) * torch.ones_like(targets), tasks=[t], num_samples=num_samples)
                output = outputs[t]

                # Accuracy
                probs = F.softmax(output, dim=2).mean(dim=0)
                _, pred = probs.max(1)
                hits = (pred == targets).float()

                # Sum
                total_hits += hits.sum().data.cpu().numpy().item()

            return total_hits / len(r)
