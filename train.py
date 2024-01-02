import os
import math
import argparse
import random

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

import data_load as data_load
# import TestModel.ViT.data_load_ffvt as data_load
# from CNN.TransFG.modeling import VisionTransformer, CONFIGS
# from CNN.VIT.modeling import VisionTransformer, CONFIGS
# from CNN.VIT.ori_vit import VisionTransformer, CONFIGS
# from CNN.EFF_Trans_a_eff.model_final import VisionTransformer, CONFIGS
# from CNN.EFF_Trans_a_eff.model_new import VisionTransformer, CONFIGS
from MARS_model.c_AAM import VisionTransformer, CONFIGS
# from CNN.EFF_Trans_b.model_b_x import VisionTransformer, CONFIGS
# from CNN.EFF_Trans_c.model_c import VisionTransformer, CONFIGS
# from CNN.SIM_Trans.modeling import VisionTransformer, CONFIGS
# from CNN.FFVT.FFVT import VisionTransformer, CONFIGS
# from scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils import read_split_data, train_one_epoch, evaluate


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def save_model(model):
    model_to_save = model
    model_checkpoint = os.path.join("./weights", "%s_checkpoint.bin" % "best_model")
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)

def main(args):
    best_acc = 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_loader, val_loader = data_load.data_process(448, 448, args.batch_size)
    model = VisionTransformer(CONFIGS['ViT-B_16'], 448, zero_head=True, num_classes=args.num_classes)
    # model = vit_base_patch16_224_in21k(num_classes=args.num_classes)

    num_params = count_parameters(model)
    print("Total Parameter: \t{:.1f}" .format(num_params))

    model.load_from(np.load(args.pretrained_dir))
    model.to(device)

    # if args.pretrained_dir != "":
    #     assert os.path.exists(args.pretrained_dir), "weights file: '{}' not exist.".format(args.pretrained_dir)
    #     weights_dict = torch.load(args.pretrained_dir, map_location=device)["model"]
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))
    # 继续训练
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)
        model.load_state_dict(pretrained_model, strict=False)
    # model.to(device)


    # model.to(args.device)
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # 余弦退火学习率优化器
    t_total = args.num_steps
    # warm_up是预热步数（步数就是总图片/batchsize），当bs=16，warm_up为500，基本代表第二个epoch开始有学习率
    # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    # set_seed(args)

    for epoch in range(args.epochs):
        # train
        train_one_epoch(model=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        device=device,
                        epoch=epoch)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model)
            # torch.save(model.state_dict(), "./weights/best_model.pth")
            # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        print("best_val_acc:{:.3f}".format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument("--pretrained_dir", type=str, default="ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    # 继续训练路径
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    #
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    # 衰减类型——余弦衰减
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    opt = parser.parse_args()

    # set_seed(opt)
    main(opt)
