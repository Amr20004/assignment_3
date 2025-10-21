import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import lib.utils.misc as misc
import time


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    # helper function
    def _format_time(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02} hours"

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()
        print(self.epoch)
        print(loader.training)

        for i, data in enumerate(loader, 1):
            # print("start")
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            torch.cuda.synchronize()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)


    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.last_log_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    # def _print_stats(self, i, loader, batch_size):
    #     self.num_frames += batch_size
    #     current_time = time.time()
    #     batch_fps = batch_size / (current_time - self.prev_time)
    #     average_fps = self.num_frames / (current_time - self.start_time)
    #     self.prev_time = current_time
    #     if i % self.settings.print_interval == 0 or i == loader.__len__():
    #         print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
    #         print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
    #         for name, val in self.stats[loader.name].items():
    #             if (self.settings.print_stats is None or name in self.settings.print_stats):
    #                 if hasattr(val, 'avg'):
    #                     print_str += '%s: %.5f  ,  ' % (name, val.avg)
    #                 # else:
    #                 #     print_str += '%s: %r  ,  ' % (name, val)
    #
    #         print(print_str[:-5])
    #         log_str = print_str[:-5] + '\n'
    #         if misc.is_main_process():
    #             print(self.settings.log_file)
    #             with open(self.settings.log_file, 'a') as f:
    #                 f.write(log_str)
    # DELETE the old _print_stats function and REPLACE it with this
    # ----------------------------------------------------------------
    def _print_stats(self, i, loader, batch_size):
        if self.settings.local_rank not in [-1, 0]:
            return

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            # 1. --- Time Calculation ---
            current_time = time.time()
            time_for_last_batch = current_time - self.last_log_time
            time_since_beginning = current_time - self.start_time

            # A more stable way to calculate time left
            avg_time_per_sample = time_since_beginning / i
            time_left_to_finish = avg_time_per_sample * (loader.__len__() - i)

            # 2. --- Time Formatting ---
            time_last_50_str = self._format_time(time_for_last_batch)
            time_beginning_str = self._format_time(time_since_beginning)
            time_left_str = self._format_time(time_left_to_finish)

            # 3. --- Build Strings ---
            time_stats_str = (f"Epoch {self.epoch} : {i} / {loader.__len__()} samples , "
                              f"time for last {self.settings.print_interval} samples : {time_last_50_str} , "
                              f"time since beginning : {time_beginning_str} , "
                              f"time left to finish the epoch : {time_left_str}")

            loss_str = "Loss values (training): {:.5f}".format(self.stats[loader.name]['Loss/total'].avg)
            iou_str = "IoU results: {:.5f}".format(self.stats[loader.name]['IoU'].avg)

            # 4. --- Print and Log ---
            print(time_stats_str)
            print(loss_str)
            print(iou_str)

            if self.settings.log_file:
                with open(self.settings.log_file, 'a') as f:
                    f.write(f"{time_stats_str}\n")
                    f.write(f"{loss_str}\n")
                    f.write(f"{iou_str}\n")

            # 5. --- Update last log time ---
            self.last_log_time = current_time

    # ----------------------------------------------------------------

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
