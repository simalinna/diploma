import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.imagenet import ImageNet

from tqdm import tqdm
import argparse

import json

from model import VICReg
import augmentations as aug



def calc_accuracy(model, linear, device, dataloader):

    total = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for images, labels in dataloader:
    
            images, labels = images.to(device), labels.to(device)
            embeddings = model.encoder(images)
            predictions = linear(embeddings)

            # Top-1
            _, predicted_top1 = torch.max(predictions.data, 1)
            correct_top1 += (predicted_top1 == labels).sum().item()

            # Top-5
            _, predicted_top5 = predictions.topk(5, 1, True, True)
            predicted_top5 = predicted_top5.t()
            correct = predicted_top5.eq(labels.view(1, -1).expand_as(predicted_top5))
            correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            
            total += labels.size(0)
    
    my_accuracy_top1=100 * correct_top1 / total
    my_accuracy_top5=100 * correct_top5 / total

    return my_accuracy_top1, my_accuracy_top5



def get_arguments():
    
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--load-exp", type=str, default="none")

    return parser

    

def main(args):

    # загружаем модель из файла и переносим на gpu
    
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    load_exp_path = '../experiments/' + args.load_exp
    state = torch.load(load_exp_path + "/model.pt", weights_only=True, map_location=device)
    num_classes = 10
    encoder_dim, projector_dim, num_classes = state["encoder_dim"], state["projector_dim"], state["num_classes"]
    model = VICReg(encoder_dim, projector_dim, num_classes).to(device)
    model.load_state_dict(state["model_state_dict"])

    # переводим модель в режим тестирования
    
    model.eval()

    # создаем однослойный линейный классификатор и переносим на gpu
    
    linear_num_classes = 10
    linear = nn.Linear(encoder_dim, linear_num_classes).to(device)

    # инициализируем аугментации
    
    transforms = aug.OneAugmentation() 
    
    # скачиваем тренировочный и тестовый датасет

    print("Загрузка датасетов...")
    
    train_data = CIFAR10(root="../datasets", train=True, download=True, transform=transforms)
    test_data = CIFAR10(root="../datasets", train=False, download=True, transform=transforms)
    
    # train_data = CIFAR100(root="./Datasets", train=True, download=True, transform=transforms)
    # test_data = CIFAR100(root="./Datasets", train=False, download=True, transform=transforms)
    
    # train_data = ImageNet(root="../Datasets", split='train', transform=transforms)
    # test_data = ImageNet(root="../Datasets", split='val', transform=transforms)

    # формируем пакеты
    
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_data, batch_size, shuffle=False, num_workers = 1)
    test_dataloader = DataLoader(test_data, batch_size, num_workers = 1)

    # инициализируем оптимизатор, планировщик и гиперпармаметры
    
    opt = Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    scheduler = StepLR(opt, step_size=10, gamma=0.3)
    num_epochs = args.epochs
    
    # инициализируем функцию ошибки классификатора
    
    criterion = nn.CrossEntropyLoss()

    print(f"\nОбучение линейного классификатора на {num_epochs} эпохах:\n")
    progress = tqdm(range(num_epochs))

    # создаем список для построения графиков
    stats_list = list()

    # создаем папку для сохранения текущего эксперимента
    
    save_exp_path = '../experiments/' + args.load_exp
    # os.makedirs(save_exp_path, exist_ok=True)
    
   # обучаем классификатор и сохраняем веса после каждой эпохи
    
    for epoch in progress:
        linear.train()
        avg_loss = 0
        for step, (images, labels) in enumerate(train_dataloader, start=epoch * len(train_dataloader)):

            images, labels = images.to(device), labels.to(device)
            embeddings = model.encoder(images)
            predictions = linear(embeddings)
            
            loss = criterion(predictions, labels)
            avg_loss += loss.item()
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            progress.set_description(f"Ошибка {loss.item():.2f}, Шаг {step+1}/{len(train_dataloader)*num_epochs}")

        scheduler.step()

        train_accuracy_top1, train_accuracy_top5 = calc_accuracy(model, linear, device, train_dataloader)
        test_accuracy_top1, test_accuracy_top5 = calc_accuracy(model, linear, device, test_dataloader)

        # сохраняем значения функции ошибки в словарь
        stats = dict(
            epoch=epoch+1,
            loss=avg_loss/len(train_dataloader),
            train_accuracy_top1 =  train_accuracy_top1,
            train_accuracy_top5 =  train_accuracy_top5,
            test_accuracy_top1 =  test_accuracy_top1,
            test_accuracy_top5 =  test_accuracy_top5,
        )

        # добавляем словарь в список
        stats_list.append(stats) 
        # сохраняем список в файл
        stats_file = open(save_exp_path + "/eval_stats.json", "w")
        print(json.dumps(stats_list), file=stats_file) 

    my_accuracy_top1, my_accuracy_top5 = calc_accuracy(model, linear, device, test_dataloader)
    
    print(f"\nТочность Top-1: {my_accuracy_top1:.2f} %")
    print(f"Точность Top-5: {my_accuracy_top5:.2f} %\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_arguments()])
    args = parser.parse_args()
    main(args)