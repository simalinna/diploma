import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import Adam

from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.imagenet import ImageNet

from tqdm.auto import tqdm
import argparse

import json

from model import VICReg
import augmentations as aug
from dataloader import MyCustomDataLoader
from loss import vicreg_loss, CE_loss



def get_arguments():
    
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--projector-dim", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=100)

    parser.add_argument("--load-exp", type=str, default="none")
    parser.add_argument("--save-exp", type=str, default="exp")

    return parser



def main(args):

    # создаем модель и переносим на gpu
    
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    encoder_dim, projector_dim, num_classes = args.encoder_dim, args.projector_dim, args.num_classes
    model = VICReg(encoder_dim, projector_dim, num_classes).to(device)
    
    # инициализируем аугментации
                
    transforms = aug.TwoAugmentations()
    
    # скачиваем тренировочный датасет
    
    print("Загрузка датасета...")
    # data = CIFAR10(root="../datasets", train=True, download=True, transform=transforms)
    data = CIFAR100(root="../datasets", train=True, download=True, transform=transforms)
    # data = ImageNet(root="../datasets", split='train', transform=transforms)

    # разделение на классы

    selected_classes = list(range(0, 70))
    indices = [i for i, (_, target) in enumerate(data) if target in selected_classes]
    data = Subset(data, indices)

    # разбиваем датасет на train и val
    
    train_size = int(len(data) * 0.9)
    val_size = int(len(data) - train_size)
    train_data, val_data = random_split(data, [train_size, val_size])

    # формируем пакеты
    
    batch_size = args.batch_size
    train_dataloader = MyCustomDataLoader(train_data, batch_size, shuffle=False)
    val_dataloader = MyCustomDataLoader(val_data, batch_size, shuffle=False)
    # train_dataloader = DataLoader(train_data, batch_size, shuffle=False)
    # val_dataloader = DataLoader(val_data, batch_size, shuffle=False)

    # инициализируем оптимизатор, планировщик и гиперпармаметры
    
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.8)
    num_epochs = args.epochs
    
    # проверяем, нужно ли загружать данные из эксперимента

    if args.load_exp != "none":
        load_exp_path = '../experiments/' + args.load_exp

        # проверяем, существует ли эксперимент
        if os.path.exists(load_exp_path) and len(os.listdir(load_exp_path))!=0:
            # загружаем данные эксперимента
            cp = torch.load(load_exp_path + "/model.pt", weights_only=True)
            model.load_state_dict(cp["model_state_dict"])
            opt.load_state_dict(cp["optimizer_state_dict"])
            scheduler.load_state_dict(cp["scheduler_state_dict"])

            # загружаем список для построения графиков
            stats_file = open(load_exp_path + "/stats.json", "r")
            stats_list = json.load(stats_file)

            epoch = cp["epoch"]
            print(f"\nМодель уже была обучена на {epoch}/{num_epochs} эпохах.\nДанные загружены из файла.\n")

            print(f"Дообучение модели на {num_epochs-epoch} эпохах:\n")
            progress = tqdm(range(cp["epoch"], num_epochs))
            
        else:
            print(f"\nЭксперимента {args.load_exp} не существует.\n")
            return 0
            
    else:
        # создаем список для построения графиков
        stats_list = list()
        
        print(f"\nОбучение модели на {num_epochs} эпохах:\n")
        progress = tqdm(range(num_epochs))

    # создаем папку для сохранения текущего эксперимента
    
    save_exp_path = '../experiments/' + args.save_exp
    os.makedirs(save_exp_path, exist_ok=True)

    # обучаем модель и сохраняем веса после каждой эпохи
    
    for epoch in progress:

        # обучение
        model.train()
        avg_loss = 0
        avg_vicreg_loss = 0
        avg_ce_loss = 0
        for step, (images, labels) in enumerate(train_dataloader, start=epoch * len(train_dataloader)):

            # x1, x2 = [x.to(device) for x in images]
            x1, x2, x3 = [x.to(device) for x in images]
            labels = labels.to(device)
            
            # rep1, rep2, pred1, pred2 = model.forward(x1, x2)
            rep1, rep2, rep3, pred1, pred2, pred3 = model.forward(x1, x2, x3)

            vicreg = vicreg_loss(rep1, rep2, rep3)
            ce = CE_loss(pred1, pred2, pred3, labels)
            loss = vicreg + ce

            avg_loss += loss.item()
            avg_vicreg_loss += vicreg.item()
            avg_ce_loss += ce.item()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            progress.set_description(f"Ошибка {loss.item():.2f}, Шаг {step+1}/{len(train_dataloader)*num_epochs}")

        # валидация
        model.eval()
        avg_val_loss = 0
        avg_val_vicreg_loss = 0
        avg_val_ce_loss = 0
        with torch.no_grad():
            for step, (images, labels) in enumerate(val_dataloader, start=epoch * len(val_dataloader)):
                
                # x1, x2 = [x.to(device) for x in images]
                x1, x2, x3 = [x.to(device) for x in images]
                labels = labels.to(device)

                rep1, rep2, rep3, pred1, pred2, pred3 = model.forward(x1, x2, x3)
                
                val_vicreg = vicreg_loss(rep1, rep2, rep3)
                val_ce = CE_loss(pred1, pred2, pred3, labels)
                val_loss = val_vicreg + val_ce

                avg_val_loss += val_loss.item()
                avg_val_vicreg_loss += val_vicreg.item()
                avg_val_ce_loss += val_ce.item()

        scheduler.step()

        # сохраняем веса модели и данные обучения в файл
        torch.save({
            "epoch": epoch + 1,
            "encoder_dim": encoder_dim,
            "projector_dim": projector_dim,
            "num_classes": num_classes,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, save_exp_path + "/model.pt")

        # вычисляем средние значения всех функций ошибки
        avg_loss = avg_loss/len(train_dataloader)
        avg_vicreg_loss = avg_vicreg_loss/len(train_dataloader)
        avg_ce_loss = avg_ce_loss/len(train_dataloader)
        avg_val_loss = avg_val_loss/len(val_dataloader)
        avg_val_vicreg_loss = avg_val_vicreg_loss/len(val_dataloader)
        avg_val_ce_loss = avg_val_ce_loss/len(val_dataloader)

        # сохраняем значения функций ошибки в словарь
        stats = dict(
            epoch=epoch+1,
            loss=avg_loss,
            vicreg_loss=avg_vicreg_loss,
            ce_loss=avg_ce_loss,
            val_loss=avg_val_loss,
            val_vicreg_loss=avg_val_vicreg_loss,
            val_ce_loss = avg_val_ce_loss,
        )

        # добавляем словарь в список
        stats_list.append(stats) 
        # сохраняем список в файл
        stats_file = open(save_exp_path + "/stats.json", "w")
        print(json.dumps(stats_list), file=stats_file) 
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_arguments()])
    args = parser.parse_args()
    main(args)