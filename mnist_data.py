from torchvision import datasets, transforms
import torch

# 前処理の定義
transform = transforms.Compose([
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize((0.1307,), (0.3081,))  # MNISTの標準化パラメータ
])

# 訓練データのダウンロードと準備
train_dataset = datasets.MNIST(
    root='./data',  # データの保存先
    train=True,  # 訓練データを取得
    download=True,  # データがなければダウンロード
    transform=transform  # 前処理を適用
)

# テストデータのダウンロードと準備
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# データローダーの設定
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
