# 2D Rendered Images - STAR Body Models

このディレクトリには、STARモデルから生成された3Dメッシュを前面・側面から撮像した2D画像が含まれています。

## 生成された画像

### 個別ビュー（前面・側面）
各体型の前面・側面画像：
- `average_front.png` / `average_side.png` - 平均体型（174.7cm）
- `tall_thin_front.png` / `tall_thin_side.png` - 高身長・痩せ型（195.0cm）
- `short_heavy_front.png` / `short_heavy_side.png` - 低身長・がっちり型（153.3cm）

### 比較ビュー（前面＋側面）
複数ビューを並べた比較画像：
- `average_front_side.png` - 平均体型の前面・側面
- `tall_and_thin_front_side.png` - 高身長・痩せ型の前面・側面
- `short_and_heavy_front_side.png` - 低身長・がっちり型の前面・側面

### グリッド比較
- `comparison_front_side.png` - 全体型の前面・側面を一覧表示

### レガシー画像（前面・背面）
以前生成した前面・背面ビュー：
- `*_back.png` - 背面ビュー
- `comparison_grid.png` - 前面・背面グリッド

## レンダリング設定

- **解像度**: 512×512 ピクセル（個別ビュー）、384×384（グリッド）
- **カメラ距離**: 3.0メートル
- **焦点距離**: 50mm相当
- **カメラ位置**:
  - **前面**: (0, 0, -3m) - Z軸負方向から人物正面を撮影
  - **側面**: (3m, 0, 0) - X軸正方向から人物右側を撮影（Y軸周り90°）
  - 背面: (0, 0, 3m) - Z軸正方向から人物背面を撮影（Y軸周り180°）

## 主な変更点

### ✨ 最新版：前面 + 側面
- より実用的なビューの組み合わせ
- 体型の奥行き（厚み）が確認可能
- 人体計測での標準的なビュー構成

### 従来版：前面 + 背面
- 仕様書（spec1.md）の合成データ生成に準拠
- Dual-view法線マップベース復元法向け

## 使用方法

これらの画像は仕様書（`md_files/spec1.md`）に記載されている合成データ生成プロセスとして生成されています。

### Python APIでの生成

```python
from models.star_layer import STARLayer
from visualizations.renderer import MeshRenderer
import torch

# STARモデル初期化
star = STARLayer(gender='neutral', num_betas=10)
betas = torch.randn(1, 10) * 0.5  # ランダムな体型

# メッシュ生成
vertices, joints = star(betas)
faces = star.get_faces()

# レンダラー作成
renderer = MeshRenderer(image_size=512, camera_distance=3.0)

# 前面・側面を生成
front_img, side_img = renderer.render_front_side(
    vertices, faces,
    save_prefix="outputs/renders/my_avatar"
)
# → my_avatar_front.png, my_avatar_side.png

# または前面・背面を生成
front_img, back_img = renderer.render_front_back(
    vertices, faces,
    save_prefix="outputs/renders/my_avatar_fb"
)
# → my_avatar_fb_front.png, my_avatar_fb_back.png

# 比較ビューを作成（複数ビュー指定可能）
renderer.render_multi_view_figure(
    vertices, faces,
    title="My Avatar",
    views=['front', 'side'],  # または ['front', 'back', 'left', 'right']
    save_path="outputs/renders/my_avatar_views.png"
)
```

## 将来の活用

将来的には以下のパイプラインが構築される予定：
1. これらの2D画像を入力として使用
2. Sapiensで法線マップ・深度マップ・ポーズ情報を抽出
3. 形状推定ネットワークでβパラメータを逆推定
4. 元の3Dモデルを再構築

---

*最終更新: 2025-11-22*
