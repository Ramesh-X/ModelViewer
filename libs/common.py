'''
Created on Jun 18, 2018

@author: rameshpr
'''

image_path = '/home/rameshpr/Documents/datasets/Market-1501-v15.09.15/query'

class FEATURE_TYPE:
    GLOBAL_FEATURES = 0
    LOCAL_FEATURES = 1
    EXP_FEATURES = 2
    MANHATTON = 3
    
class MODEL_TYPE:
    RESNET50 = 'resnet50'
    DENSENET161 = 'densenet161'
    
class Sizes:
    querry_x = 250
    querry_y = 250
    image_x = 80
    image_y = 80
    n_rows = 12
    n_cols = 18
    
model_locs = {
    MODEL_TYPE.RESNET50 : './models/resnet50.h5',
    MODEL_TYPE.DENSENET161 : './models/densenet161.h5'
}
