import cv2
import torch

from gui import Ui_MainWindow
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QDialog, QApplication
from models import cifarresnet18
import json
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.Testimage = None
        self.setupUi(self)  # 设置界面
        self.pushButton.clicked.connect(self.upload_image)  # 绑定点击信号和槽函数
        self.pushButton_2.clicked.connect(self.classifier) # 绑定点击信号和槽函数
        self.resnet = torch.load('./resnet18_entire_model.pth')
        self.resnet.eval()
        self.target = []
        with open('label.json', 'r') as json_file:
            # 从文件中加载数据
            datas = json.load(json_file)
            for index in range(len(datas)):
                k = str(index)
                self.target.append(datas[k][1])

    def upload_image(self):  # click对应的槽函数
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.jpeg)",
                                                   options=options)
        if file_name:
            pixmap = QtGui.QPixmap(file_name)
            self.Testimage = cv2.imread(file_name)
            scaled_pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)
            self.label.setScaledContents(True)  # 使标签内的内容自适应标签大小

    def classifier(self):
        self.label_3.setText("识别中...")

        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        preprocess = transforms.Compose([transforms.ToPILImage(),transforms.Resize(32),transforms.ToTensor(),normalize,])

        pytorch_tensor = preprocess(self.Testimage)
        pytorch_tensor = pytorch_tensor.to(torch.device("cuda:0"))
        with torch.no_grad():
            outputs = self.resnet(pytorch_tensor.unsqueeze(0))
            _, predicted = torch.max(outputs[1], 1)
        self.label_3.setText(self.target[predicted.item()])

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyApp()
    myapp.show()
    sys.exit(app.exec_())
