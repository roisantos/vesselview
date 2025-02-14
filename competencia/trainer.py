import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
from utils.helpers import dir_exists, get_instance, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta


class Trainer:
    def __init__(self, model, loss, CFG, train_loader, val_loader, device):
        super(Trainer, self).__init__()
        self.device = device
        self.loss = loss
        self.CFG = CFG
        self.train_loader = train_loader  # Correctly initialize train_loader
        self.val_loader = val_loader
        self.test_loader = None  # Initialize test_loader to None

        # Use bracket notation for CFG access
        if self.CFG['amp'] is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.model = model.to(self.device)
        # self.model.load_state_dict(checkpoint['state_dict'])  # This line is removed

        if device == 'cuda':
            cudnn.benchmark = True

        self.optimizer = get_instance(
            torch.optim, "optimizer", CFG, self.model.parameters())
        self.lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        # start_time = datetime.now().strftime('%y%m%d%H%M%S')
        self.checkpoint_dir = os.path.join(
            CFG['save_dir'], CFG['dataset']['type'], CFG['loss']['type']) # Note: changed the directory structure because of the OOD training.
        dir_exists(self.checkpoint_dir)
        self.writer = tensorboard.SummaryWriter(log_dir=self.checkpoint_dir) # Correctly initialize writer

    def train(self):
        for epoch in range(1, self.CFG['epochs'] + 1):  # Use bracket notation
            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.CFG['val_per_epochs'] == 0:  # Use bracket notation
                results = self._valid_epoch(epoch)
                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'{str(k):15s}: {v}')
            if epoch % self.CFG['save_period'] == 0: # adjust to best model # Use bracket notation
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        wrt_mode = 'train'
        self._reset_metrics()
        tbar = tqdm(self.train_loader, total=len(self.train_loader))
        tic = time.time()
        for img, gt in tbar:
            self.data_time.update(time.time() - tic)
            img = img.to(self.device) #cuda(non_blocking=True) #TODO: Hopefully figure out why this is not working.
            gt = gt.to(self.device) #cuda(non_blocking=True)
            self.optimizer.zero_grad()

            if self.CFG['amp'] is True:  # Use bracket notation
                with torch.cuda.amp.autocast(enabled=True):
                    pre = self.model(img)
                    if isinstance(pre, tuple): # Wnet model returns a tuple in trainning mode.
                        logits_aux, logits = pre
                        loss_aux = self.loss(logits_aux, gt)
                        loss = loss_aux + self.loss(logits, gt)
                    else:
                        loss = self.loss(pre, gt)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pre = self.model(img)
                if isinstance(pre, tuple): # Wnet model returns a tuple in trainning mode.
                        logits_aux, logits = pre
                        loss_aux = self.loss(logits_aux, gt)
                        loss = loss_aux + self.loss(logits, gt)
                else:
                        loss = self.loss(pre, gt)
                loss.backward()
                self.optimizer.step()


            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)
            if isinstance(pre, tuple):
                pre = pre[1] # for evaluation, select the last output.
            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG['threshold']).values())  # Use bracket notation
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} MCC {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
            tic = time.time()
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for img, gt in tbar: # type: ignore
                img = img.to(self.device) #.cuda(non_blocking=True)
                gt = gt.to(self.device) #.cuda(non_blocking=True)
                if self.CFG['amp'] is True:  # Use bracket notation
                    with torch.autocast(device_type='cuda', enabled=True):
                        predict = self.model(img)
                        if isinstance(predict, tuple): # Wnet model returns a tuple in trainning mode.
                            logits_aux, logits = predict
                            loss_aux = self.loss(logits_aux, gt)
                            loss = loss_aux + self.loss(logits, gt)
                        else:
                            loss = self.loss(predict, gt)
                else:
                    predict = self.model(img)
                    loss = self.loss(predict, gt)
                self.total_loss.update(loss.item())

                if isinstance(predict, tuple):
                    predict = predict[1] # for evaluation, select the last output.
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG['threshold']).values())  # Use bracket notation
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} MCC {:.4f}|'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))
                self.writer.add_scalar(
                    f'{wrt_mode}/loss', self.total_loss.average, epoch)

        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG  # Save the entire config (as a dictionary)
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.batch_time.update(0, weight=1)
        
        self.data_time = AverageMeter()
        self.data_time.update(0, weight=1)
        
        self.total_loss = AverageMeter()
        self.total_loss.update(0, weight=1)
        
        self.auc = AverageMeter()
        self.auc.update(0, weight=1)
        
        self.f1 = AverageMeter()
        self.f1.update(0, weight=1)
        
        self.acc = AverageMeter()
        self.acc.update(0, weight=1)
        
        self.sen = AverageMeter()
        self.sen.update(0, weight=1)
        
        self.spe = AverageMeter()
        self.spe.update(0, weight=1)
        
        self.pre = AverageMeter()
        self.pre.update(0, weight=1)
        
        self.iou = AverageMeter()
        self.iou.update(0, weight=1)
        
        self.mcc = AverageMeter()
        self.mcc.update(0, weight=1)
        
        self.CCC = AverageMeter()
        self.CCC.update(0, weight=1)


    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou, mcc):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)
        self.mcc.update(mcc)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average,
            "MCC": self.mcc.average
        }
