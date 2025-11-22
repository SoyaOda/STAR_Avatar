# STAR Avatar System

3D人体形状推定システム - 画像からSTAR体型パラメータを推定し、3Dアバターを生成

## プロジェクト構成

```
STAR_Avatar/
├── models/              # STARモデル実装
│   └── star_layer.py   # STAR PyTorchレイヤー
├── setup/              # セットアップスクリプト
│   ├── requirements.txt
│   └── download_star_model.py
├── tests/              # テストスクリプト
│   └── test_star_generation.py
├── visualizations/     # 可視化ツール
│   └── mesh_viewer.py
├── data/               # データディレクトリ
│   └── star_models/    # STAR .npz モデルファイル (要ダウンロード)
├── outputs/            # 生成結果の出力先
└── md_files/           # 仕様書・計画書
    ├── general_spec.md
    ├── spec1.md
    └── implementation_plan_first_step.md
```

## クイックスタート

### 1. 依存パッケージのインストール

```bash
# Python 3.8+ 推奨
pip install -r setup/requirements.txt
```

### 2. STAR体型モデルで3Dメッシュ生成テスト

```bash
# 基本テスト（最小実装版）
python tests/test_star_generation.py
```

このテストでは：
- ランダムなβパラメータから3D人体メッシュを生成
- 複数の体型バリエーションを生成
- Matplotlib/Open3Dで可視化
- OBJ形式でメッシュを保存

### 3. 公式STARモデルのダウンロード（オプション）

より正確な体型生成には、公式STARモデルが必要です：

1. https://star.is.tue.mpg.de/ にアクセス
2. アカウント登録
3. 以下をダウンロード：
   - `STAR_NEUTRAL.npz` (Gender-neutral)
   - `STAR_MALE.npz` (Male)
   - `STAR_FEMALE.npz` (Female)
4. `data/star_models/` に配置

```bash
# ヘルプスクリプト
python setup/download_star_model.py
```

## 使用方法

### 基本的なメッシュ生成

```python
from models.star_layer import STARLayer
import torch

# STARモデル初期化
star = STARLayer(gender='neutral', num_betas=10)

# 体型パラメータ（β）
betas = torch.randn(1, 10) * 0.5  # ランダムな体型

# メッシュ生成
vertices, joints = star(betas)

# vertices: [1, 6890, 3] - 頂点座標
# joints: [1, 24, 3] - 関節位置
```

### 3Dメッシュ可視化

```python
from visualizations.mesh_viewer import visualize_mesh_open3d, save_mesh_obj

# インタラクティブ3Dビューア
faces = star.get_faces()
visualize_mesh_open3d(vertices, faces)

# OBJファイルに保存
save_mesh_obj(vertices, faces, 'outputs/my_avatar.obj')
```

### 2D画像レンダリング（前面・背面）

```python
from visualizations.renderer import MeshRenderer

# レンダラー初期化
renderer = MeshRenderer(
    image_size=512,        # 出力解像度
    camera_distance=3.0,   # カメラ距離（メートル）
    focal_length=50.0      # 焦点距離（mm）
)

# 前面・背面画像を生成
front_img, back_img = renderer.render_front_back(
    vertices, faces,
    save_prefix="outputs/renders/my_avatar"
)
# 保存: my_avatar_front.png, my_avatar_back.png

# 比較ビューを作成
renderer.render_multi_view_figure(
    vertices, faces,
    title="My Avatar",
    save_path="outputs/renders/my_avatar_views.png"
)
```

## 実装状況

### ✅ Phase 1.1-1.2: STARコア実装（完了）
- [x] STARレイヤー（PyTorch）
- [x] βパラメータ → メッシュ生成
- [x] バッチ処理対応
- [x] 3Dメッシュ可視化（Open3D, Matplotlib）
- [x] **2D画像レンダリング（前面・背面ビュー）** ✨ NEW
- [x] テストスクリプト

### 🚧 Phase 1.3-1.6: 形状推定ネットワーク（次ステップ）
- [ ] Sapiens前処理パイプライン
- [ ] ResNet18ベース形状推定CNN
- [ ] 合成データ生成
- [ ] 学習パイプライン
- [ ] REST APIサーバー

### 📋 Phase 2: Unity統合（計画中）
- [ ] Unityプロジェクトセットアップ
- [ ] βパラメータ適用スクリプト (C#)
- [ ] APIクライアント統合

詳細は `md_files/implementation_plan_first_step.md` を参照。

## 技術スタック

- **PyTorch** 2.0+ - 深層学習フレームワーク
- **NumPy** - 数値計算
- **Open3D** - 3Dメッシュ可視化
- **Matplotlib** - プロット・可視化
- **Trimesh** - メッシュ処理

## テスト内容

`test_star_generation.py` では以下をテスト：

1. **基本メッシュ生成** - β=0（平均体型）
2. **体型バリエーション** - 主成分ごとの変化
3. **バッチ生成** - 複数メッシュの一括生成
4. **平行移動** - translation パラメータ
5. **関節可視化** - 24関節の位置表示
6. **インタラクティブビューア** - Open3D 3Dビューア

## トラブルシューティング

### Open3Dが起動しない
```bash
# Matplotlib版を使用（ヘッドレス環境対応）
# test_star_generation.py内で自動フォールバック
```

### 公式STARモデルがない
```bash
# 最小実装版（簡易円柱メッシュ）で動作確認可能
# 精度が必要な場合は公式モデルをダウンロード
```

## 参考資料

- [STAR公式サイト](https://star.is.tue.mpg.de/)
- [STAR論文 (ECCV 2020)](https://arxiv.org/abs/2008.08535)
- [GitHub: ahmedosman/STAR](https://github.com/ahmedosman/STAR)
- [SMPL Unity実装ガイド](https://files.is.tue.mpg.de/nmahmood/smpl_website/How-to_SMPLinUnity.pdf)

## ライセンス

このプロジェクトはデモ・研究目的です。STARモデル自体のライセンスは公式サイトを参照してください。

---

*作成日: 2025-11-21*
