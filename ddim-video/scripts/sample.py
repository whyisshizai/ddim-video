import argparse
import torch
import datetime
import torchvision
import numpy as np
from PIL import Image
from vddpm import script_util
import cv2
def create_argparser():
    device = torch.device("cuda")
    defaults = dict(
        video_length=16,
        video_size=(32,32),
        attention_resolutions=(0, 1, 1, 1),
        num_timesteps=2000,
        fps = 4,
        num_images=1,
        eta = 0.02,
        time_step =1000,
        name= "Anime",
        use_labels=None,
        device=device,
        #控制噪声的调度
        schedule_low=0.1,  #low
        schedule_high=0.5,  #high
        model_path = r"D:\pycharm\open-cv\ddpm-video\model\28_anime_video-video-2025-04-09-00-17-iteration-1000-model.pth",
        save_dir = r"D:\pycharm\open-cv\ddpm-video\results",
    )
    defaults.update(script_util.diffusion_defaults())

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str)
    # parser.add_argument("--save_dir", type=str)
    script_util.add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    # args.num_timesteps = 1000
    args.video_length=16
    device = args.device
    try:
        diffusion = script_util.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path), strict=False)
        diffusion.eval()
        # 更改一下参数
        diffusion.num_timesteps = args.time_step
        # diffusion.video_size = (24,24)
        args.use_labels = [1,2,4,8,10,12,15,20,25] #传进去对不同参数进行生成
        #使用图片进行生成
        args.use_img = None
        # args.use_img = r'D:\pycharm\open-cv\DDPM-main\results\img\cat-2024-12-09-19-05-0-0.5.png'
        # args.large_time = 2

        print("start")
        s = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        if args.use_img:
            print('using image',args.use_img)
            image = np.array(Image.open(args.use_img))
            height, width, channels = image.shape
            # 创建一个新的矩阵，尺寸为原始图像的两倍，填充为高斯噪声
            new_height, new_width = height * args.large_time, width * args.large_time
            # mean 0,std为0.1,控制噪声强度
            mean,std = 0,0.1
            noise = np.random.normal(mean, std, (new_height, new_width, channels)).astype(image.dtype)
            # 将噪声限制在图像的有效范围内（假设是0-255的范围）
            noise = np.clip(noise, 0, 255)
            # 用噪声初始化新矩阵
            new_matrix = noise
            #填充
            new_matrix[::2, ::2] = image
            transform = script_util.get_transform()
            #第0维添加一个维度
            x = transform(new_matrix).unsqueeze(0)
            x = x.to(device)
            diffusion.video_size = (new_height, new_width)
            print('larging....................')
            #args.num_images // 10
            samples = diffusion.sample(batch_size=1, device = device, x=x)
            print('图片扩大中')
            print("sample:(",samples.size())
            for batch_idx in range(samples.shape[0]):
                video_tensor = samples[batch_idx]
                video_tensor = ((video_tensor + 1) / 2).clip(0, 1)
                video_tensor = video_tensor.permute(1, 2, 3, 0)  # [t, h, w, c]
                video_tensor = (video_tensor * 255).byte().cpu().numpy()

                output_filename = f"{args.save_dir}/ddim-{args.name}-{s}-{batch_idx}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'XVID', 'MJPG' 等
                height, width = video_tensor.shape[1], video_tensor.shape[2]
                out = cv2.VideoWriter(output_filename, fourcc, args.fps, (width, height))

                for frame in video_tensor:
                    out.write(frame)

                out.release()
                print(f"视频 {batch_idx} 保存完毕: {output_filename}")
        else:
            if args.use_labels:
                print("use labels",args.use_labels)
                args.num_images *= 10
                for label in args.use_labels:
                    print('processing....................')
                    samples = diffusion.implicit_sample(args.num_images // 10, device, eta = args.eta,prompt=label)
                    print('处理结束')
                    for batch_idx in range(samples.shape[0]):
                        video_tensor = samples[batch_idx]
                        video_tensor = ((video_tensor + 1) / 2).clip(0, 1)
                        video_tensor = video_tensor.permute(1, 2, 3, 0)  # [t, h, w, c]
                        video_tensor = (video_tensor * 255).byte().cpu().numpy()
                        output_filename = f"{args.save_dir}/ddim-{label}-{args.name}-{s}-{args.time_step}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或者 'XVID', 'MJPG' 等
                        height, width = video_tensor.shape[1], video_tensor.shape[2]
                        out = cv2.VideoWriter(output_filename, fourcc, args.fps, (width, height))
                        for frame in video_tensor:
                            out.write(frame)
                            out.write(frame)
                        out.release()
                        print(f"视频 {batch_idx} 保存完毕: {output_filename}")
            else:
                print(f'not use {args.use_labels}')
                samples = diffusion.implicit_sample(args.num_images, device,time_steps=args.time_step,eta = args.eta)
                print("sample:(",samples.size())
                for batch_idx in range(samples.shape[0]):
                    video_tensor = samples[batch_idx]
                    video_tensor = ((video_tensor + 1) / 2).clip(0, 1)
                    video_tensor = video_tensor.permute(1, 2, 3, 0)  # [t, h, w, c]
                    video_tensor = (video_tensor * 255).byte().cpu().numpy()

                    output_filename = f"{args.save_dir}/ddim-{args.name}-{s}-{batch_idx}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'XVID', 'MJPG' 等
                    height, width = video_tensor.shape[1], video_tensor.shape[2]
                    out = cv2.VideoWriter(output_filename, fourcc, args.fps, (width, height))

                    for frame in video_tensor:
                        out.write(frame)

                    out.release()
                    print(f"视频 {batch_idx} 保存完毕: {output_filename}")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")