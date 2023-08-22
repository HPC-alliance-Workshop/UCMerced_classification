import os
from torch.utils.data import DataLoader, Dataset
import torch
from skimage import io
import os
import argparse
import random
import matplotlib.pyplot as plt
from torchvision.transforms import transforms


class UCMerced(Dataset) :
    def __init__(self, dataset_path, device, is_train=True) -> None:
        super().__init__()


        random_seed = 42
        random.seed(random_seed)

        self.images_path = os.path.join(dataset_path, "Images")
        self.device = device
        self.is_train = is_train
        self.train_transform =  transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.RandomVerticalFlip()
        ])
        self.val_transform =  transforms.Compose([
                transforms.CenterCrop(224)
        ])


        train_size = int(100 * 0.8)
        val_size = int(100 * 0.2)


        self.class_names = ['agricultural' , 'baseballdiamond' , 'buildings' , 'denseresidential' , 'freeway'  ,   'harbor'   , 'mediumresidential' , 'overpass'  ,  'river' ,
                              'sparseresidential' , 'tenniscourt' , 'airplane'  ,    'beach'    ,   'chaparral' ,  'forest'    , 
                                'golfcourse' ,  'intersection' ,  'mobilehomepark' ,  'parkinglot' , 'runway' , 'storagetanks']
        
    
        self.num_classes = len(self.class_names)

        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        
        self.train_images = []
        self.val_images = []


        for class_name in self.class_names :
            images = os.listdir(os.path.join(self.images_path, class_name))
            images = [os.path.join(self.images_path, class_name, item) for item in images]
            train_sample = random.sample(images, train_size)
            
            val_sample = [item for item in images if item not in train_sample]

            self.train_images += train_sample
            self.val_images += val_sample


        if self.is_train :
            print("Size of Training dataset ", len(self.train_images))
        else :
            print("Size of validation dataset  ", len(self.val_images))
            


    def __len__(self) :
        return len(self.train_images) if self.is_train else len(self.val_images)


    def __getitem__(self, idx) :

        if self.is_train :
            image_path = self.train_images[idx]

        else :
            image_path = self.val_images[idx]


        image = io.imread(image_path)
        image = torch.tensor(image).float()
        image = image.permute(2, 0, 1)


        class_idx = self.class_to_index[image_path.split('/')[-2]]

        if self.is_train :
            image = self.train_transform(image)
        else :
            image = self.val_transform(image)

        
        #label = torch.nn.functional.one_hot(torch.tensor(class_idx), self.num_classes)
        label = torch.tensor(class_idx)

        image = image.to(self.device)
        label = label.to(self.device)

        

        return image, label
        








        











    


















if __name__ == "__main__" :


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    ds = UCMerced(dataset_path=args.dataset_path, device=device)


    ds[0]


