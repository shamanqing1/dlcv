import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dlcv.data.pascal_voc import VOCDetection



if __name__ == '__main__':
    trainset = VOCDetection(root='~/.datasets/VOCdevkit', splits=((2007, 'trainval'),))

    img, label = trainset[0]
    print(img.shape, label)