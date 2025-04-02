import random
import torch
import numpy as np


class MyCustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
        if self.shuffle:
            import random
            random.shuffle(self.indices)

    def __iter__(self):
        # Итерация по батчам
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            images, labels = zip(*[self.dataset[idx]  for idx in batch_indices])
            num_outs = len(images[0])
            if (num_outs == 1):
                images = torch.stack(images)
            elif (num_outs == 2):
                x, y = zip(*[image for image in images])
                x, y = torch.stack(x), torch.stack(y)
            labels = torch.tensor(labels)
            pairs = []
            for j in range(len(batch_indices)):
                
                # Выбираем случайным образом два изображения из одного класса
            
                same_indices = np.where(labels.numpy() == labels[j].numpy())[0]
                same_indices = np.delete(same_indices, np.where(same_indices == j)[0])
                
                random_index = np.random.choice(same_indices, 1)[0]
                pair = x[random_index] if num_outs == 2 else images[random_index]
    
                pairs.append(pair)
                
            pairs = torch.stack(pairs)

            if num_outs == 2:
                data = [x, y, pairs]
            elif num_outs == 1:
                data = [images, pairs]

            yield data, labels

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # Количество батчей