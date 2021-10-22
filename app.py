

from array import array
import os

import sys
import time
import json

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTConfig
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import streamlit as st
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

transform = transforms.Compose([ # 検証データ用の画像の前処理
    transforms.RandomResizedCrop(size=(224,224),scale=(1.0,1.0),ratio=(1.0,1.0)), # アスペクト比を保って画像をリサイズ
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class Data(Dataset):
    def __init__(self, img):
        self.images = []
        self.categories = []

        # データの定義
        try:
            # 学習用の処理を行なっていないデータ
            feature_ids = torch.reshape(transform(img),(-1, 3, 224, 224))
            self.images.append(feature_ids)
        except:
            print('error')
            pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

# モデルクラスの宣言

configuration = ViTConfig()
vit_model = ViTModel(configuration)

class ViTNet(nn.Module):
    def __init__(self, pretrained_vit_model, class_num):
        super(ViTNet, self).__init__()
        self.vit = vit_model
        self.fc = nn.Linear(768, class_num)

    def _get_cls_vec(self, states):
        return states['last_hidden_state'][:, 0, :]

    def forward(self, input_ids):
        states = self.vit(input_ids)
        states = self._get_cls_vec(states)
        states = self.fc(states)
        return states

st.title('はんだ判定アプリ')

#st.write("画像を分類する数を指定してください")
#class_num  = st.slider('分類数', 1, 10, 4)

class_num = 4

net = ViTNet(ViTModel, class_num)
model_path = './model/model.pth'
net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

#upload_model = st.file_uploader('AIモデルをアップロードしてください',type=['pth'])

#if upload_model is not None:
#    net.load_state_dict(torch.load(upload_model,map_location=torch.device('cpu')))
#else:
#    model_path = './model/model0.pth'
#    net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

uploaded_file = st.file_uploader('検査する写真をアップロードが撮影してください。', type=['jpg','png','jpeg'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_path = f'img/{uploaded_file.name}'
    img.save(img_path)
    
    data = Data(img)

    # DataLoaderを取得する
    loader = DataLoader(data, batch_size=1, shuffle=False)

    for batch in loader:
        with torch.no_grad():
            input_ids = batch[0]
            out = net(input_ids)
            predict = out.argmax(dim=1)


    st.markdown('認識結果')
    if predict.detach().numpy()[0] == 0:
        st.title("正常")
    else:
        st.title("不良")
        st.write("不良の種別")
        st.write(predict)
        st.write(predict.detach().numpy()[0])
        st.write("1:ブリッジ 2:角 3:芋")

    st.image(img)
