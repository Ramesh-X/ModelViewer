from __future__ import absolute_import, print_function, division
'''
Created on Jun 18, 2018

@author: rameshpr
'''
from PyQt4 import QtCore
from keras import backend as K
import numpy as np
import cv2

from models import densenet_no_top, resnet50_notop
from .common import model_locs, MODEL_TYPE
import tensorflow as tf

class FeatureExtractor(QtCore.QThread):


    def __init__(self, model_type, median=0):
        QtCore.QThread.__init__(self)
        self.image_path = None
        self.model_type = model_type
        self.median=median
        self.outputs = None
        self._create_model()
        self.model.model._make_predict_function()
        self.graph = tf.get_default_graph()
        
    def __del__(self):
        self.wait()
        
    def set_image(self, image_path):
        self.image_path = image_path
        
    def _create_model(self):
        weight_path = model_locs[self.model_type]
        if self.model_type == MODEL_TYPE.DENSENET161:
            self.model = densenet_no_top(weight_path=weight_path)
        elif self.model_type == MODEL_TYPE.RESNET50:
            self.model = resnet50_notop(weight_path=weight_path)
        else:
            raise ValueError("Not implemented yet")

    def run(self):
        with self.graph.as_default():
            inp = self.model.model.input                                           # input placeholder
            outputs = [layer.output for layer in self.model.model.layers]          # all layer outputs
            functor = K.function([inp], outputs) # evaluation function
            
            # Testing
            test = cv2.imread(self.image_path)
            self.input_image = np.copy(test)
            if self.model_type == MODEL_TYPE.DENSENET161:
                test = self.model.preprocess(test)
            elif self.model_type == MODEL_TYPE.RESNET50:
                test = self.model.preprocess(test, True, False)
            else:
                raise ValueError("Not implemented yet for the model, %s" % self.model_type)
            test = np.expand_dims(test, axis=0)
            layer_outs = functor([test, 1.])
        self.outputs = layer_outs
