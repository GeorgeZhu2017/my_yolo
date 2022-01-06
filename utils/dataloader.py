import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor

class YoloDataset(Dataset):
    def __init__(self, dataset_path, splits, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.dataset_path = dataset_path
        self.splits = splits
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.length = len(self.splits)
        self.train = train

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        split = self.splits[index]
        image, box = self.get_data(split)
        image = np.transpose((np.array(image, dtype=np.float32) / 255.0), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box                

    def get_data(self, split, jitter=.3):
        image = Image.open(self.dataset_path + 'images/' + split + '.png')
        image = cvtColor(image)
        iw, ih  = image.size
        h, w    = self.input_shape
        box = np.array(self.get_box(split))

        if self.train:
            new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.25, 2)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw,nh), Image.BICUBIC)
            dx = int(self.rand(0, w-nw))
            dy = int(self.rand(0, h-nh))
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = new_image
            if self.rand()<.5: 
                image_data = image_data.transpose(Image.FLIP_LEFT_RIGHT)            
        else:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            # add padding stripe
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

        # adjust bboxes
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        return image_data, box
                    

    def get_box(self, split):
        with open(self.dataset_path + 'annotations/' + split+ '.csv', 'r') as file:
            annotations = np.loadtxt(file, delimiter = ',', skiprows = 1).reshape(-1, 5)
        if len(annotations) == 0:
            return []

        boxes = np.zeros([len(annotations), 5])
        for i, anno in enumerate(annotations):
            boxes[i, 0] = anno[2] - anno[3]
            boxes[i, 1] = anno[1] - anno[3]
            boxes[i, 2] = anno[2] + anno[3]
            boxes[i, 3] = anno[1] + anno[3]  
            boxes[i, 4] = 0
        return boxes      

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes
