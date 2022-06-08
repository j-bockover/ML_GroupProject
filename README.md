# ML_GroupProject
CS 545 Machine Learning Group Project for Spring 2022
## Collaborators
Jason Bockover  
Minwei Luo  
Arpankumar Rajpurohit  
Alvin Iskender  

## Dataset:  
The dataset used for this project is from:
https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification?resource=download&select=images 

## Usage

Download and extract the dataset files into directory '/data', run model(s) with command:

    python main.py [a choice of VGG or Resnet]

### optional arguments

|arguments|definition|
|----|----|
|`--lr`|learning rate|
|`--epochs`|number of training epochs for the model|
|`--batch_size`|batch size|
|`--weight_decay`|weight decay value set for Adam optimizer|

### e.g.

    python main.py VGG --lr 0.001 --epochs 100 --batch_size 32 --weight_decay 1e-6
