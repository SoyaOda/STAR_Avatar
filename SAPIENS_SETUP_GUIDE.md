#  Sapiens完全セットアップガイド

## 📋 現在の状況

✅ **完了済み**:
- Sapiensリポジトリのクローン
- Python依存関係のインストール
- ディレクトリ構造の作成
- 推論ラッパー (`inference/sapiens_wrapper.py`) の作成

❌ **未完了**:
- Sapiensモデルのダウンロード（大容量ファイル）

---

## 🚀 モデルダウンロード手順

### 方法1: Git LFS（推奨）

```bash
# 1. Git LFSのインストール（未インストールの場合）
brew install git-lfs
git lfs install

# 2. チェックポイントディレクトリへ移動
export SAPIENS_CHECKPOINT=/Users/moei/program/sapiens_lite_host/torchscript

# 3. Normal Estimation モデル (0.3B) をダウンロード
mkdir -p $SAPIENS_CHECKPOINT/normal/checkpoints/sapiens_0.3b
cd $SAPIENS_CHECKPOINT/normal/checkpoints/sapiens_0.3b
git clone https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript .

# 4. Depth Estimation モデル
mkdir -p $SAPIENS_CHECKPOINT/depth/checkpoints/sapiens_0.3b
cd $SAPIENS_CHECKPOINT/depth/checkpoints/sapiens_0.3b
git clone https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript .

# 5. Pose Estimation モデル
mkdir -p $SAPIENS_CHECKPOINT/pose/checkpoints/sapiens_0.3b
cd $SAPIENS_CHECKPOINT/pose/checkpoints/sapiens_0.3b
git clone https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript .

# 6. Segmentation モデル
mkdir -p $SAPIENS_CHECKPOINT/seg/checkpoints/sapiens_0.3b
cd $SAPIENS_CHECKPOINT/seg/checkpoints/sapiens_0.3b
git clone https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript .
```

### 方法2: 手動ダウンロード

以下のリンクから `.pt2` ファイルを直接ダウンロード：

1. **Normal Estimation (法線推定)**
   - URL: https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript/tree/main
   - ファイル: `sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2`
   - 保存先: `/Users/moei/program/sapiens_lite_host/torchscript/normal/checkpoints/sapiens_0.3b/`

2. **Depth Estimation (深度推定)**
   - URL: https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript/tree/main
   - ファイル: `sapiens_0.3b_render_people_epoch_88_torchscript.pt2`
   - 保存先: `/Users/moei/program/sapiens_lite_host/torchscript/depth/checkpoints/sapiens_0.3b/`

3. **Pose Estimation (姿勢推定)**
   - URL: https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/tree/main
   - ファイル: `sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2`
   - 保存先: `/Users/moei/program/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_0.3b/`

4. **Segmentation (セグメンテーション)**
   - URL: https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript/tree/main
   - ファイル: `sapiens_0.3b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2`
   - 保存先: `/Users/moei/program/sapiens_lite_host/torchscript/seg/checkpoints/sapiens_0.3b/`

---

## 🔧 使用方法

### Option 1: GTデータ生成（実際のSapiensなし）

```bash
# フォトリアリスティック画像から対応するGTデータを生成
python generate_sapiens_style_outputs.py
```

**出力**:
- `outputs/sapiens_style/average_front_normal.png`
- `outputs/sapiens_style/average_front_depth.png`
- `outputs/sapiens_style/average_front_mask.png`
- `outputs/sapiens_style/average_front_joints_heatmap.png`

### Option 2: 実際のSapiens推論（モデルダウンロード後）

```python
from inference.sapiens_wrapper import SapiensInference

# 実際のSapiensモデルを使用
sapiens = SapiensInference(model_size='0.3b', use_mock=False)

# 推論実行
results = sapiens.infer(
    image_path='outputs/renders/average_photorealistic_front.png',
    output_dir='outputs/sapiens_inference'
)
```

---

## 📊 モデルサイズと選択

| モデル | パラメータ数 | 推論速度 | 精度 | 推奨用途 |
|--------|------------|----------|------|----------|
| 0.3B | 300M | 最速 | 良 | リアルタイム、プロトタイプ |
| 0.6B | 600M | 速い | より良 | バランス型 |
| 1B | 1000M | 中速 | 高精度 | 高品質出力 |
| 2B | 2000M | 遅い | 最高精度 | オフライン処理 |

**推奨**: まず0.3Bで試し、精度が必要なら1Bに変更

---

## ✅ セットアップ確認

```bash
# モデルが正しくダウンロードされたか確認
ls -lh /Users/moei/program/sapiens_lite_host/torchscript/*/checkpoints/sapiens_0.3b/*.pt2

# 推論ラッパーのテスト
python inference/sapiens_wrapper.py
```

---

## 🐛 トラブルシューティング

### エラー: "No checkpoint found"

**原因**: モデルファイルが正しい場所にない

**解決策**:
1. ディレクトリ構造を確認
2. `.pt2` ファイルが存在するか確認
3. ファイル名が期待通りか確認

### エラー: "Failed to load model"

**原因**: PyTorchバージョンの不一致

**解決策**:
```bash
python3 -m pip install torch>=2.2.0
```

### Git LFS のエラー

**原因**: Git LFSがインストールされていない

**解決策**:
```bash
brew install git-lfs
git lfs install
```

---

## 📚 参考資料

- **公式GitHub**: https://github.com/facebookresearch/sapiens
- **論文**: https://arxiv.org/abs/2408.12569
- **HuggingFace**: https://huggingface.co/facebook/sapiens
- **Lite版README**: `/Users/moei/program/sapiens/lite/README.md`

---

## 💡 次のステップ

1. ✅ モデルをダウンロード
2. ✅ `sapiens_wrapper.py` で `use_mock=False` に変更
3. ✅ フォトリアリスティック画像で推論テスト
4. ✅ 学習パイプラインに統合

---

**作成日**: 2025-11-22
**最終更新**: 2025-11-22
