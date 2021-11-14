import torch
from torchvision import transforms
from torchvision.models import mobilenetv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import streamlit as st
from PIL import Image

# GPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([ # 検証データ用の画像の前処理
    #transforms.RandomResizedCrop(size=(224,224),scale=(1.0,1.0),ratio=(1.0,1.0)), # アスペクト比を保って画像をリサイズ
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# モデルクラスの宣言

mob_model = mobilenetv2.mobilenet_v2(pretrained=False)

class Mobilenetv2(nn.Module):
    def __init__(self, mob_model, class_num):
        super(Mobilenetv2, self).__init__()
        self.vit = mob_model
        self.fc = nn.Linear(1000, class_num)

    def forward(self, input_ids):
        states = self.vit(input_ids)
        states = self.fc(states)
        return states

st.title('はんだ判定アプリ')

#st.write("画像を分類する数を指定してください")
#class_num  = st.slider('分類数', 1, 10, 4)

class_num = 4

uploaded_file = st.file_uploader('検査する写真をアップロードが撮影してください。', type=['jpg','png','jpeg'])
if uploaded_file is not None:
    net = None

    img = Image.open(uploaded_file)

    data = torch.reshape(transform(img),(-1,3,224,224))

    net = Mobilenetv2(mob_model, class_num)
    model_path = './model/model.pth'
    net.load_state_dict(torch.load(model_path,map_location=device))

    net.eval()

    with torch.no_grad():
        out = net(data)
        predict = out.argmax(dim=1)
        #st.write(out)

    syubetsu = ["ブリッジ","角","芋"]

    st.markdown('認識結果')
    if predict.detach().numpy()[0] == 0:
        st.title("正常")
    else:
        st.title("不良")
        st.write("不良の種別")
        #st.write(predict.detach().numpy()[0])
        st.write(syubetsu[predict.detach().numpy()[0] - 1])

    st.image(img)
