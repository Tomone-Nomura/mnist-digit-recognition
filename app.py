import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
from mnist_model import Net
import random

# ページ設定
st.set_page_config(page_title="手書き数字認識アプリ", layout="wide")

# タイトルと説明
st.title("手書き数字認識アプリ")
st.write("このアプリは手書きの数字（0〜9）を認識します。画像をアップロードするか、サンプル画像を使用してください。")

# モデルのロード
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# MNISTテストデータセットのロード
@st.cache_resource
def load_test_dataset():
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    return test_dataset

test_dataset = load_test_dataset()

# 画像の前処理
def preprocess_image(image):
    # グレースケールに変換
    if image.mode != 'L':
        image = image.convert('L')
    
    # リサイズ（MNISTは28x28）
    image = image.resize((28, 28))
    
    # 前処理（テンソル変換と正規化）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(image).unsqueeze(0)  # バッチ次元を追加

# テンソルをPIL画像に変換
def tensor_to_pil(tensor):
    # テンソルをnumpyに変換
    img = tensor.squeeze().numpy()
    
    # [0, 1]の範囲にスケーリング
    img = (img * 255).astype(np.uint8)
    
    # PIL画像に変換
    return Image.fromarray(img)

# 予測関数
def predict(image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)[0]
        prediction = output.argmax(dim=1, keepdim=True).item()
    
    return prediction, probabilities.numpy()

# サイドバー - アップロード方法の選択
option = st.sidebar.selectbox(
    "画像入力方法を選択",
    ["画像をアップロード", "サンプル画像を使用"]
)

# メイン領域を2列に分割
col1, col2 = st.columns(2)

with col1:
    if option == "画像をアップロード":
        uploaded_file = st.file_uploader("手書き数字の画像をアップロード", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="アップロードされた画像", width=300)
            
            # 予測ボタン
            if st.button("予測する"):
                prediction, probabilities = predict(image)
                
                with col2:
                    st.subheader(f"予測結果: {prediction}")
                    
                    # 確率のバーチャート表示
                    fig, ax = plt.subplots(figsize=(8, 4))
                    digits = np.arange(10)
                    ax.bar(digits, probabilities, color='skyblue')
                    ax.set_xlabel('数字')
                    ax.set_ylabel('確率')
                    ax.set_title('予測確率')
                    ax.set_xticks(digits)
                    st.pyplot(fig)
                    
                    # 前処理後の画像表示
                    processed_tensor = preprocess_image(image)
                    processed_image = processed_tensor.squeeze().numpy()
                    st.subheader("前処理後の画像")
                    st.image(processed_image, width=150, clamp=True)
    
    else:  # サンプル画像を使用
        st.subheader("サンプル画像")
        st.write("MNISTテストセットからランダムに画像を表示します。「新しいサンプル」ボタンで別の画像を表示できます。")
        
        # ランダムなインデックスを生成
        if 'random_idx' not in st.session_state:
            st.session_state.random_idx = random.randint(0, len(test_dataset)-1)
        
        # 新しいサンプルボタン
        if st.button("新しいサンプル"):
            st.session_state.random_idx = random.randint(0, len(test_dataset)-1)
        
        # サンプル画像の取得と表示
        sample_img, sample_label = test_dataset[st.session_state.random_idx]
        sample_pil = tensor_to_pil(sample_img)
        st.image(sample_pil, caption=f"サンプル画像（正解: {sample_label}）", width=300)
        
        # 予測ボタン
        if st.button("予測する（サンプル）"):
            prediction, probabilities = predict(sample_pil)
            
            with col2:
                st.subheader(f"予測結果: {prediction}")
                st.write(f"正解: {sample_label}")
                
                # 予測が正しいかどうかを表示
                if prediction == sample_label:
                    st.success("正解！")
                else:
                    st.error("不正解...")
                
                # 確率のバーチャート表示
                fig, ax = plt.subplots(figsize=(8, 4))
                digits = np.arange(10)
                ax.bar(digits, probabilities, color='skyblue')
                ax.set_xlabel('数字')
                ax.set_ylabel('確率')
                ax.set_title('予測確率')
                ax.set_xticks(digits)
                st.pyplot(fig)

# アプリについての追加情報
st.sidebar.markdown("---")
st.sidebar.subheader("アプリについて")
st.sidebar.write("このアプリは手書き数字を認識するためのCNNモデルを使用しています。MNISTデータセットで訓練され、98%以上の精度を達成しています。")
st.sidebar.write("作成者: Tomone　Nomura")