# -*- coding: utf-8 -*-
import gc
import time

import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data 
#这里使用多线程
import torch.optim as optim
import vddpm.script_util as script_utils
import wandb

import torchvision
import argparse
import datetime
import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import vddpm.script_util as script_utils


class RescaleChannels(object):
    def __call__(self, sample):
        return 2 * sample - 1
    
def get_transform():

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Use a consistent port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()

class VideoDataset(Dataset):
    def __init__(self, file, transform=None):
        # 存储图片文件和标签
        self.file = file
        self.video_files = []
        self.prompt = []
        self.length = []
        # login("fill your hugging face access token here")
        i = 0
        #控制一下dataset的长度
        for filename in os.listdir(file)[:]:
            print(filename)
            if filename.endswith("mp4"):
                i+=1
                #注意RGBA格式
                name = filename.split(".")[0]
                cap = cv2.VideoCapture(os.path.join(file, filename))
                # video = mpy.VideoFileClip(f"{file}/{filename}")
                
                frames = []
                prt = int(name)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    f =transform(frame)
                    f = f.unsqueeze(0).permute(1,0,2,3)
                    frames.append(f)
                #真实的长度
                s = len(frames)
                while s >= 8:
                    self.video_files.append(torch.cat(frames[s-8:s],dim=1))
                    self.prompt.append(prt)
                    s-=8
                if s >= 2:
                    self.video_files.append(torch.cat(frames[0:8], dim=1))
                    self.prompt.append(prt)
                #[c,t,h,w]
        # self.image_files = torch.from_numpy(np.array(self.image_files))
        self.transform = transform
    def __len__(self):
        return len(self.video_files)
    def __getitem__(self, idx):
        return self.video_files[idx], torch.tensor(0), self.prompt[idx]



def create_argparser():
    device = torch.device("cuda:1")
    run_name = datetime.datetime.now().strftime("video-%d-%H-%M")
    defaults = dict(
        file = r'D:\mizunashi akari\video\32',
        activation = 'mish',
        use_ddim = True,
        learning_rate=1e-5,
        batch_size=6,
        iterations=80000,
        video_size=(32,32),
        video_length=8,
        num_timesteps = 2000,
        epoch = 1,
        log_to_wandb=False,
        log_rate=1,
        checkpoint_rate=500,
        attention_resolutions = (0,1,1,1),
        log_dir=r"D:\pycharm\open-cv\ddpm-video\model",
        project_name='28_anime_video',
        run_name=run_name,
        model_checkpoint=r"D:\pycharm\open-cv\ddpm-video\model\28_anime_video-video-2025-04-09-00-17-iteration-1000-model.pth",
        optim_checkpoint=r"D:\pycharm\open-cv\ddpm-video\model\28_anime_video-video-2025-04-09-00-17-iteration-1000-optim.pth",
        #设定好的
        schedule_low=1e-5,
        #DDP
        ip='tcp://127.0.0.1', 
        port = '20000',
        # world_size = 4,
        # rank=0,
        schedule_high=0.02,
        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())
    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world_size', type=int, help='Number of processes.')
    return parser


def train(rank, world_size, args,dataset):
    
    #setup(rank, world_size)
    print(f"cuda:{rank}正在启动")

    #分发到对应的GPU上面
    device = torch.device(f"cuda:{rank}")

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = optim.Adamax(diffusion.parameters(), lr=args.learning_rate)
        print(f"cuda:{rank}模型已经建立")

        #DDP
        #diffusion = DDP(diffusion, device_ids=[rank])
        print(f"model已经分发到cuda:{rank}")

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint, map_location=device))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")
            run = wandb.init(
                project=args.project_name,
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        #dataset and DistributedSampler
        #sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            # sampler=sampler,
            # num_workers=args.num_workers,
            # pin_memory=True,  # Important for faster data transfer to GPU
            #无所谓我们bs = 1
            drop_last=True, # Drop last incomplete batch
        )
        print(len(dataloader))
        print(f"cuda:{rank}开始训练")
        # Training loop
        for iteration in range(1, args.iterations + 1):
            diffusion.train()
            # sampler.set_epoch(iteration)

            for i, (x, y, prompt) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                prompt = prompt.to(device)
                # l = l.to(device)

                optimizer.zero_grad()
                if args.use_labels:
                    loss = diffusion(x, y, prompt)
                else:
                    loss = diffusion(x, prompt)

                loss.backward()
                optimizer.step()

                # # Update EMA (if you have one; assuming diffusion has update_ema)
                # if hasattr(diffusion.module, 'update_ema'):
                diffusion.update_ema()

            if iteration % args.log_rate == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()},time:{datetime.datetime.now().strftime("%H:%M:%S")}")
                if args.log_to_wandb:
                    wandb.log({
                            "train_loss": loss.item(),
                        })

            if iteration % args.checkpoint_rate == 0 and rank == 0:
                model_filename = os.path.join(args.log_dir, f"{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth")
                optim_filename = os.path.join(args.log_dir, f"{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth")
                torch.save(diffusion.state_dict(), model_filename)  # Save the underlying model, not the DDP wrapper
                torch.save(optimizer.state_dict(), optim_filename)

        cleanup()

        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        # if args.log_to_wandb:
        #     run.finish()
        print("Keyboard interrupt,程序提前结束")


if __name__ == "__main__":
    args = create_argparser().parse_args()
    world_size = torch.cuda.device_count()  #现有的GPU个数
    args.run_name = datetime.datetime.now().strftime("video-%Y-%m-%d-%H-%M") # Set run name here
    
    dataset = VideoDataset(args.file, transform=get_transform())
    print("数据集已经建立")


    gc.collect()
    
    train(0,4,args,dataset)
    #我们启用四个线程
    # mp.spawn(train,
    #          args=(world_size, args, dataset),
    #          nprocs=world_size,
    #          join=True)