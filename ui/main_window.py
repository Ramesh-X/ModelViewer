'''
Created on Jun 18, 2018

@author: rameshpr
'''
from PyQt4 import QtCore, QtGui
import os
import cv2
import numpy as np

from lib import FeatureExtractor
from lib.common import image_path, Sizes, MODEL_TYPE


class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.i = 0
        self.page_i = 0
        self.layer_i = 0
        self.__outs = None
        self.__thread = FeatureExtractor(MODEL_TYPE.DENSENET161, median=0)
        self.__thread.finished.connect(self.__finished)
        self.images = [os.path.join(image_path, f) for f in os.listdir(image_path)]
        self.images = [f for f in self.images if f.endswith(".jpg") and os.path.isfile(f)]
        self.__setup()
        
    def __load_features(self):
        if self.__thread.isRunning():
            return
        self.__thread.set_image(self.images[self.i])
        self.__thread.start()
        
    def __next_img_button_clicked(self):
        self.setEnabled(False)
        self.i += 1
        self.__load_features()
        
    def __prev_img_button_clicked(self):
        self.setEnabled(False)
        self.i -= 1
        if self.i < 0:
            self.i = 0
        self.__load_features()
        
    def __layer_combobox_changed(self, index):
        self.setEnabled(False)
        self.layer_i = int(index)
        self.show_images()
    
    def __prev_sheet_button_clicked(self):
        self.setEnabled(False)
        self.page_i -= 1
        if self.page_i < 0:
            self.page_i = 0
        self.show_images()
    
    def __next_sheet_button_clicked(self):
        self.setEnabled(False)
        self.page_i += 1
        self.show_images()
        
    def __setup(self):
        self.__load_features()
        button_panel = QtGui.QWidget(self)
        image_panel = QtGui.QWidget(self)
        
        self.__qurryLabel = QtGui.QLabel(button_panel)
        self.__qurryLabel.resize(Sizes.querry_x, Sizes.querry_y)
        self.__qurryLabel.setScaledContents(True)
        self.__qurryLabel.move(10, 15)
        
        self.__prevImgButton = QtGui.QPushButton("Prev Image",button_panel)
        self.__prevImgButton.clicked.connect(self.__prev_img_button_clicked)
        self.__prevImgButton.resize(100, 25)
        self.__prevImgButton.move(10, Sizes.querry_y + 25)
        
        self.__nextImgButton = QtGui.QPushButton("Next Image",button_panel)
        self.__nextImgButton.clicked.connect(self.__next_img_button_clicked)
        self.__nextImgButton.resize(100, 25)
        self.__nextImgButton.move(113, Sizes.querry_y + 25) 
               
        self.__selectLayerLabel = QtGui.QLabel("Select Layer",button_panel)
        self.__selectLayerLabel.adjustSize()
        self.__selectLayerLabel.move(10, Sizes.querry_y + 60)
        
        self.__layerComboBox = QtGui.QComboBox(button_panel)
        self.__layerComboBox.currentIndexChanged.connect(self.__layer_combobox_changed)
        self.__layerComboBox.resize(100, 25)
        self.__layerComboBox.move(113, Sizes.querry_y + 55)
        
        self.__prevSheetButton = QtGui.QPushButton("Prev Sheet",button_panel)
        self.__prevSheetButton.clicked.connect(self.__prev_sheet_button_clicked)
        self.__prevSheetButton.resize(100, 25)
        self.__prevSheetButton.move(10, Sizes.querry_y + 85)
        
        self.__nextSheetButton = QtGui.QPushButton("Next Sheet",button_panel)
        self.__nextSheetButton.clicked.connect(self.__next_sheet_button_clicked)
        self.__nextSheetButton.resize(100, 25)
        self.__nextSheetButton.move(113, Sizes.querry_y + 85)
        
        self.__imageLabels = []
        for i in range(Sizes.n_cols * Sizes.n_rows):
            self.__imageLabels.append(QtGui.QLabel(image_panel))
            self.__imageLabels[i].setScaledContents(True)
            self.__imageLabels[i].resize(Sizes.image_x, Sizes.image_y)
            x = i % Sizes.n_cols
            y = int(i/Sizes.n_cols)
            self.__imageLabels[i].move(3*(x+1) + Sizes.image_x*x, 3*(y+1) + Sizes.image_y*y)
        
        self.__infoLabel = QtGui.QLabel("", self)
        self.__infoLabel.adjustSize()
        
        image_panel.resize(Sizes.n_cols*(Sizes.image_x+3), Sizes.n_rows*(Sizes.image_y+3))
        button_panel.resize(max(Sizes.querry_x+20, 223), max(Sizes.querry_y + 110, image_panel.height()))
        button_panel.move(0, 0)
        self.__infoLabel.move(button_panel.width(), 0)
        image_panel.move(button_panel.width(), 15)
        self.setEnabled(False)
        self.resize(image_panel.width() + button_panel.width() + 10, max(image_panel.height() + 20, button_panel.height()) + 10)
    
    def show_images(self):
        self.__infoLabel.setText("Image %s ::> Layer: %d and page: %d" % (self.images[self.i].split('/')[-1], self.layer_i, self.page_i))
        self.__infoLabel.adjustSize()
        for image_label in self.__imageLabels:
            image_label.clear()
        limit = len(self.__imageLabels)
        j = 0
        for i in range(self.page_i*limit, (self.page_i+1)*limit):
            if len(self.__outs[self.layer_i][0].shape) < 3:
                break
            if i >= self.__outs[self.layer_i][0].shape[2]:
                if i == self.page_i*limit:
                    self.page_i -= 1
                    self.show_images()
                break
            self.__imageLabels[j].setPixmap(self.convert_image(self.__outs[self.layer_i][0][:,:,i]))
            j += 1
        self.setEnabled(True)
        
    def __finished(self):
        if self.__thread is None:
            return
        self.__qurryLabel.setPixmap(self.convert_image(self.__thread.input_image))
        self.__outs = self.__thread.outputs
        if len(self.__layerComboBox) == 0:
            self.__layerComboBox.addItems([str(i) for i in range(len(self.__outs))])
        self.show_images()
        
    def convert_image (self, cv_img):
        cv_img = cv_img.astype(np.float32)
        cv_img = cv_img - np.min(cv_img)
        img_max = np.max(cv_img)
        if img_max != 0:
            cv_img = 255 * cv_img / img_max 
        cv_img = cv_img.astype(np.uint8)
        if len(cv_img.shape) == 2:
            cv_img = np.expand_dims(cv_img, axis=2)
            cv_img = np.repeat(cv_img, 3, axis=2)
        height, width, bytesPerComponent = cv_img.shape
        bytesPerLine = bytesPerComponent * width;
        cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB, cv_img)
        return QtGui.QPixmap.fromImage(QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888))



