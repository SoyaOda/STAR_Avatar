# STAR Avatar System - 実装状況

## 概要

spec1.mdの仕様に基づき、3D人体形状復元システムの合成データ生成パイプラインを実装しました。

## 実装完了項目

### 1. STARモデル統合 ✓

**ファイル**: `models/star_layer.py`

- STARモデル（Sparse Trained Articulated Human Body Regressor）のPyTorch実装
- 形状パラメータβ（10次元）による体型制御
- 6890頂点、24関節の人体メッシュ生成
- 男性/女性/ニュートラルモデル対応

**仕様書対応**:
- spec1.md 10-13行目: STARモデル実装の要件
- spec1.md 66-70行目: STARレイヤーの使用方法

### 2. PyTorchベースレンダラー ✓

**ファイル**: `visualizations/pytorch_renderer.py`

仕様書で要求される以下の出力を生成：

#### a) 法線マップ（Normal Maps）
- カメラ座標系での表面法線ベクトルをRGBエンコード
- 形式: (n+1)/2で[0,1]範囲に正規化
- 出力: [H, W, 3] RGB画像

**仕様書対応**: spec1.md 160-161行目

#### b) 深度マップ（Depth Maps）
- カメラからの距離（Z-buffer）
- 身長基準での正規化対応（デフォルト1.7m）
- 出力: [H, W] グレースケール画像

**仕様書対応**: spec1.md 162-164行目

#### c) 関節ヒートマップ（Joint Heatmaps）
- 24関節それぞれのガウシアンヒートマップ
- 標準偏差σ=5ピクセル（設定可能）
- 出力: [H, W, K] マルチチャンネル画像（K=24）

**仕様書対応**: spec1.md 165-166行目

#### d) 人物マスク（Segmentation Mask）
- 人物領域の2値マスク
- 背景=0、人物=1
- 出力: [H, W] バイナリ画像

**仕様書対応**: spec1.md 167-168行目

### 3. 仮想カメラシステム ✓

**実装内容**:
- 焦点距離: 50mm相当（視野角約30°）
- センサーサイズ: 36mm（フルフレーム）
- 画像解像度: 512×512ピクセル
- カメラ距離: 2.5～3.5m（ランダムサンプリング対応）
- カメラ位置: (Δx, Δy, -D) ※Δx, Δy = ±2%D

**カメラビュー**:
- 正面（front）: (0, 0, -D)
- 背面（back）: (0, 0, +D)、Y軸周り180°回転
- 側面（side/right）: (D, 0, 0)、Y軸周り90°回転
- 左側面（left）: (-D, 0, 0)、Y軸周り-90°回転

**仕様書対応**: spec1.md 158-159行目

### 4. 合成データ生成パイプライン ✓

**ファイル**: `generate_synthetic_data.py`

**機能**:
1. ランダムな形状パラメータβのサンプリング（標準正規分布）
2. STARモデルによる3Dメッシュ生成
3. 前面・背面からの仮想撮像
4. 法線・深度・関節・マスクの生成
5. 教師データ（βパラメータ）の保存

**出力構造**:
```
outputs/synthetic_data/
├── sample_1/
│   ├── front_normal.png        # 前面法線マップ
│   ├── front_depth.png         # 前面深度マップ
│   ├── front_joints_heatmap.png # 前面関節ヒートマップ
│   ├── front_mask.png          # 前面人物マスク
│   ├── back_normal.png         # 背面法線マップ
│   ├── back_depth.png          # 背面深度マップ
│   ├── back_joints_heatmap.png # 背面関節ヒートマップ
│   ├── back_mask.png           # 背面人物マスク
│   └── beta_gt.npy             # 教師データ（β）
├── sample_2/
└── sample_3/
```

**仕様書対応**: spec1.md 153-172行目（合成データ生成）

### 5. 可視化用レンダラー（既存）✓

**ファイル**: `visualizations/renderer.py`

- Matplotlibベースの簡易レンダラー
- RGB画像の生成
- 前面・側面・背面ビュー対応
- 検証・可視化用途

**用途**:
- 生成されたメッシュの確認
- デバッグ用ビジュアライゼーション
- ドキュメント用画像生成

### 6. フォトリアリスティックレンダラー ✓

**ファイル**: `visualizations/photorealistic_renderer.py`

- pyrenderベースのPBR（Physically-Based Rendering）レンダラー
- 写真のような高品質RGB画像の生成
- 3点照明システム（Key Light、Fill Light、Back Light）
- スムースシェーディングによる滑らかな表面表現
- 肌色マテリアル（ベージュ）のデフォルト設定

**技術仕様**:
- PBRマテリアル（Metallic=0.0、Roughness=0.7）
- 3点照明（強度3.0/1.5/1.0）+ 環境光（強度0.5）
- OpenGLベースオフスクリーンレンダリング
- 前面・背面・左右側面ビュー対応

**用途**:
- プロダクション品質の可視化
- プレゼンテーション用画像
- 写真のようなリアルな体型表示

**詳細**: `md_files/photorealistic_rendering.md`

## 仕様書との対応表

| 仕様書項目 | 実装状況 | ファイル |
|-----------|---------|---------|
| STARモデル統合 | ✓ 完了 | models/star_layer.py |
| 仮想カメラ設定 | ✓ 完了 | pytorch_renderer.py |
| 法線マップ生成 | ✓ 完了 | pytorch_renderer.py |
| 深度マップ生成 | ✓ 完了 | pytorch_renderer.py |
| 関節ヒートマップ | ✓ 完了 | pytorch_renderer.py |
| セグメンテーション | ⚠ 部分対応 | pytorch_renderer.py |
| 合成データ生成 | ✓ 完了 | generate_synthetic_data.py |
| フォトリアリスティックRGB | ✓ 完了 | photorealistic_renderer.py |
| データ拡張 | ⏸ 未実装 | - |
| 形状推定ネットワーク | ⏸ 未実装 | - |
| 学習パイプライン | ⏸ 未実装 | - |
| Sapiens統合 | ⏸ 未実装 | - |

## 技術仕様

### カメラ内部パラメータ

```python
# 焦点距離（ピクセル単位）
fx = (focal_length / sensor_width) * image_size
fy = fx

# 画像中心
cx = image_size / 2.0
cy = image_size / 2.0

# 透視投影
x_2d = fx * (X / Z) + cx
y_2d = fy * (Y / Z) + cy
```

### 深度正規化

仕様書（spec1.md 162-164行目）に従い、被写体の身長を基準に深度をスケーリング：

```python
# 現在のメッシュ身長を計算
current_height = vertices[:, 1].max() - vertices[:, 1].min()

# 目標身長（例: 1.7m）に正規化
scale = normalize_height / current_height
depth_normalized = depth * scale
```

これにより、異なる体型・カメラ距離でも統一スケールの深度マップが得られます。

### 法線エンコーディング

カメラ座標系の法線ベクトル (nx, ny, nz) を RGB に変換：

```python
R = (nx + 1) / 2  # [-1, 1] → [0, 1]
G = (ny + 1) / 2
B = (nz + 1) / 2
```

## 使用方法

### 1. 合成データ生成

```bash
python generate_synthetic_data.py
```

デフォルトで3サンプルの合成データを `outputs/synthetic_data/` に生成します。

### 2. Python APIでの使用

```python
from models.star_layer import STARLayer
from visualizations.pytorch_renderer import STARRenderer
import torch

# モデル初期化
star = STARLayer(gender='neutral', num_betas=10)
renderer = STARRenderer(image_size=512, focal_length=50.0)

# ランダムな体型生成
betas = torch.randn(1, 10) * 0.5
vertices, joints = star(betas)
faces = star.get_faces()

# レンダリング
outputs = renderer.render_all(
    vertices=vertices[0],
    faces=torch.from_numpy(faces).long(),
    joints_3d=joints[0],
    camera_distance=3.0,
    view='front',
    normalize_height=1.7
)

# 出力:
# - outputs['normal']: 法線マップ [512, 512, 3]
# - outputs['depth']: 深度マップ [512, 512]
# - outputs['joint_heatmaps']: 関節ヒートマップ [512, 512, 24]
# - outputs['mask']: 人物マスク [512, 512]
```

### 3. 可視化用レンダリング

```python
from visualizations.renderer import MeshRenderer

# RGB画像生成用
vis_renderer = MeshRenderer(image_size=512, camera_distance=3.0)
front_img, side_img = vis_renderer.render_front_side(
    vertices, faces,
    save_prefix="outputs/renders/my_avatar"
)
```

## 今後の実装予定

### Phase 1: データ生成の拡張
- [ ] データ拡張（ノイズ付加、ブラー等）
- [ ] 部位セグメンテーションの完全実装
- [ ] ポーズバリエーション（±10°肩、±5°肘等）
- [ ] 大規模合成データ生成（10,000+サンプル）

### Phase 2: 学習パイプライン
- [ ] 形状推定ネットワーク（ResNet18ベース）
- [ ] 損失関数（β損失、位置損失、幾何損失）
- [ ] 学習スクリプト（Adam、AMP対応）
- [ ] 検証パイプライン

### Phase 3: Sapiens統合
- [ ] Sapiensモデルのロード
- [ ] 法線・深度・ポーズ推論
- [ ] 前処理パイプライン統合

### Phase 4: 最適化・推論
- [ ] メッシュ位置合わせ（深度ベース）
- [ ] 最適化工程（LBFGS）
- [ ] 身体寸法計測（ウエスト、ヒップ等）
- [ ] REST API実装

## パフォーマンス

現在の実装（CPU、Python 3.9、PyTorch 2.8）:
- STARメッシュ生成: ~10ms
- 1ビューのレンダリング: ~500ms（512×512）
- 合成データ1サンプル生成: ~1秒

将来の最適化:
- GPU対応による高速化（10-100倍）
- バッチレンダリング
- C++/TorchScript化

## 参考文献

- STAR: Sparse Trained Articulated Human Body Regressor (ECCV 2020)
- STRAPS: Shape Trained with RGB and PaS (BMVC 2020)
- Sapiens: Foundation for Human Vision Models (Meta AI, 2024)

---

**最終更新**: 2025-11-22
**実装者**: Claude Code
**仕様書**: md_files/spec1.md
