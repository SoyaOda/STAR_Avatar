# STAR Avatar System - 実装計画 First Step

## プロジェクト概要

正面・背面画像から3D人体形状（STAR体型パラメータβ）を推定し、Unityでアバターとして表示するシステム

### システム構成
- **バックエンド（Python）**: 画像 → Sapiens → βパラメータ推定（spec1.md準拠）
- **フロントエンド（Unity）**: βパラメータ → 3Dアバター変形・表示

---

## 技術調査結果サマリー

### 重要な発見

1. **SMPL/STARのUnity実装は公式提供されている**
   - SMPLはUnity用FBX、C#スクリプト、実装ガイドを提供
   - STARはSMPLのドロップイン代替（パラメータ20%削減、より現実的な変形）
   - 同じアプローチで実装可能

2. **VRMとの統合は公開例なし**
   - VRMブレンドシェイプは主に表情用
   - 体型の大幅変更には向かない
   - **推奨**: 独自のSTARメッシュ使用（VRM互換性は後回し）

3. **Unityでの体型変形メカニズム**
   - Shape Blendshapes: βパラメータによる体型変化
   - Pose Blendshapes: 414個のポーズ補正（LBSアーティファクト修正）
   - Linear Blend Skinning (LBS): 4ボーンスキニング
   - ブレンドウェイト変換: SMPL(-1〜+1) → Unity(0〜1)

---

## 実装優先順位とフェーズ

### Phase 1: Pythonバックエンド（推論パイプライン）
**目的**: 画像からβパラメータを推定する最小限のAPI

#### 1.1 環境構築とデータ前処理
**実装内容**:
```
├── setup/
│   ├── environment.py          # PyTorch, PyTorch3D, OpenCV環境セットアップ
│   └── download_models.py      # STAR, Sapiensモデルダウンロード
├── preprocessing/
│   ├── sapiens_inference.py    # Sapiens推論（法線/深度/ポーズ/セグメント抽出）
│   ├── depth_normalization.py  # 深度マップ身長スケール補正
│   ├── pose_heatmap.py         # 2D関節→ヒートマップ変換
│   └── segmentation.py         # 部位セグメンテーション前処理
```

**テスト方法**:
- テスト画像（正面/背面ペア）を入力
- 出力: 法線マップ、深度マップ、Kチャネルヒートマップ、Cチャネルセグメント
- 可視化スクリプトで各マップを画像保存し、目視確認
- 深度スケールが身長（1.7m基準）に正規化されているか数値確認

**期間**: 3-4日

---

#### 1.2 形状推定ネットワーク構築
**実装内容**:
```
├── models/
│   ├── shape_estimator.py      # ResNet18ベース形状推定CNN
│   │   - 入力: 法線(3) + 深度(1) + ポーズ(K) + セグ(C) × 2視点
│   │   - 出力: β(n次元) + T(3次元)
│   ├── star_layer.py           # STARモデルPyTorchレイヤー
│   └── network_config.py       # ハイパーパラメータ設定
```

**テスト方法**:
- ランダム入力テンソルでforward pass
- 出力形状確認（β: [1, 10], T: [1, 3]）
- 勾配flow確認（backward pass）
- STARレイヤーでβ→メッシュ変換確認（頂点数6890×3）

**期間**: 2-3日

---

#### 1.3 合成データ生成（学習用）
**実装内容**:
```
├── data_generation/
│   ├── synthetic_data.py       # β, θサンプリング
│   ├── renderer.py             # PyTorch3D: メッシュ→法線/深度/セグメント
│   ├── augmentation.py         # データ拡張（ノイズ、ブラー）
│   └── dataloader.py           # PyTorch DataLoader
```

**テスト方法**:
- 100サンプル生成
- レンダリング結果を画像保存
- データ拡張前後の比較
- バッチローダーの動作確認（メモリリーク、速度測定）

**期間**: 3-4日

---

#### 1.4 学習パイプライン
**実装内容**:
```
├── training/
│   ├── train.py                # メイン学習ループ
│   ├── losses.py               # L_β, L_T, L_geo損失
│   ├── optimizer.py            # Adam, learning rate schedule
│   └── validation.py           # 検証データ評価
```

**テスト方法**:
- 小規模データ（1000サンプル）で学習
- 損失曲線の収束確認
- βMSE、ウエスト周囲長誤差（cm）を指標
- オーバーフィッティング確認（train/val loss比較）

**期間**: 4-5日

---

#### 1.5 最適化工程（オプション）
**実装内容**:
```
├── optimization/
│   ├── refine.py               # LBFGS最適化
│   └── geometry_losses.py      # L_n, L_d, L_s, L_j
```

**テスト方法**:
- 推論結果を初期値として最適化
- 最適化前後のメッシュを可視化
- 関節位置誤差（px）、身長誤差（cm）測定

**期間**: 2-3日

---

#### 1.6 REST APIサーバー
**実装内容**:
```
├── server/
│   ├── app.py                  # Flask/FastAPI
│   ├── inference_api.py        # POST /estimate エンドポイント
│   └── response_formatter.py   # JSON: {beta: [...], T: [...]}
```

**テスト方法**:
- curlでテスト画像送信
- レスポンスJSON確認
- 推論速度測定（~0.5秒目標）

**期間**: 1-2日

**Phase 1 合計**: 約2-3週間

---

### Phase 2: Unityフロントエンド（アバター表示）
**目的**: βパラメータを受け取り、3Dアバターに適用

#### 2.1 Unityプロジェクトセットアップ
**実装内容**:
```
Assets/
├── STAR_Avatar/
│   ├── Models/                 # STARメッシュFBX（SMPL公式準拠）
│   ├── Scripts/
│   │   ├── STARController.cs  # βパラメータ→ブレンドシェイプ
│   │   ├── APIClient.cs       # REST APIクライアント
│   │   └── CameraCapture.cs   # 正面/背面画像キャプチャ
│   ├── Materials/
│   └── Scenes/
│       └── MainScene.unity
```

**必要アセット**:
- STAR/SMPLメッシュFBX（公式サイトから取得）
- REST Client for Unity（または UnityWebRequest使用）

**テスト方法**:
- 空シーンにSTARメッシュ配置
- Inspector で SkinnedMeshRenderer 確認
- Hierarchy でボーン構造確認（24ジョイント）

**期間**: 1日

---

#### 2.2 βパラメータ適用スクリプト
**実装内容**:
```csharp
// STARController.cs
public class STARController : MonoBehaviour
{
    [SerializeField] SkinnedMeshRenderer meshRenderer;
    [SerializeField] int numBetas = 10;

    public void ApplyBeta(float[] beta)
    {
        // βパラメータ → Shape Blendshape係数変換
        // meshRenderer.SetBlendShapeWeight(index, weight);
    }

    void UpdatePoseBlendshapes()
    {
        // 414個のPose Blendshapes更新（SMPL公式スクリプト参考）
    }
}
```

**テスト方法**:
- 手動でβ=[1,0,0,...,0]を設定
- メッシュ変形を確認（第1主成分の効果）
- β=[0,1,0,...,0]で第2主成分確認
- ランダムβでリアルタイム変形テスト

**期間**: 2-3日

---

#### 2.3 REST APIクライアント統合
**実装内容**:
```csharp
// APIClient.cs
public class APIClient : MonoBehaviour
{
    string serverURL = "http://localhost:5000/estimate";

    public IEnumerator EstimateShape(Texture2D frontImg, Texture2D backImg)
    {
        // 画像をbase64エンコード
        // POST リクエスト送信
        // JSONレスポンス受信 → β配列をデコード
        yield return response;
    }
}
```

**テスト方法**:
- Pythonサーバー起動
- Unityからテスト画像送信
- 返されたβをSTARControllerに適用
- エンドツーエンドテスト

**期間**: 2日

---

#### 2.4 カメラキャプチャ機能
**実装内容**:
```csharp
// CameraCapture.cs
public class CameraCapture : MonoBehaviour
{
    public Texture2D CaptureFront() { /*...*/ }
    public Texture2D CaptureBack() { /*...*/ }
}
```

**テスト方法**:
- Webカメラから正面/背面撮影
- 解像度512×512確認
- 画像プレビュー表示

**期間**: 1-2日

---

#### 2.5 UIとワークフロー統合
**実装内容**:
- ボタン: "撮影" → "推定実行" → "結果表示"
- プログレスバー
- βパラメータ数値表示

**テスト方法**:
- ユーザーシナリオテスト
- エラーハンドリング確認

**期間**: 2日

**Phase 2 合計**: 約1-1.5週間

---

## First Stepで実装する最小機能セット

### 優先度A（必須）: Phase 1.1〜1.2
1. **Sapiens前処理パイプライン** (preprocessing/)
2. **形状推定ネットワーク** (models/shape_estimator.py)
3. **STARレイヤー** (models/star_layer.py)

**理由**: これがシステムのコア。まずエンドツーエンドの推論が動くことを確認

**テスト戦略**:
- `test_preprocessing.py`: Sapiens出力の形状・値域確認
- `test_network.py`: forward/backward pass確認
- `test_star.py`: β→メッシュ変換の精度確認

**成功基準**:
- テスト画像から法線/深度/ポーズ/セグメント抽出成功
- ランダムβで3Dメッシュ生成成功
- 推論速度 < 1秒

---

### 優先度B（重要）: Phase 1.3〜1.4
4. **合成データ生成** (data_generation/)
5. **学習パイプライン** (training/)

**理由**: ネットワークの精度向上に必要

**テスト戦略**:
- 小規模学習（1000サンプル、5エポック）
- 損失収束確認
- 検証データで誤差測定

**成功基準**:
- β MSE < 0.5（合成データ上）
- ウエスト周囲長誤差 < 3cm

---

### 優先度C（後回し）: Phase 1.5〜1.6, Phase 2
6. 最適化工程
7. REST APIサーバー
8. Unity統合

---

## 開発環境

### Python環境
- Python 3.8+
- PyTorch 2.x, PyTorch3D
- Sapiens（TorchScript版）
- STAR公式実装（ahmedosman/STAR）
- OpenCV, PIL
- Flask/FastAPI

### Unity環境
- Unity 2022.3 LTS推奨
- UniVRM（VRM対応時）
- REST Client for Unity（または標準UnityWebRequest）

### ハードウェア
- GPU: NVIDIA RTX 3060以上（VRAM 12GB+）
- CPU: マルチスレッド対応
- RAM: 16GB+

---

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| Sapiensモデルが大きすぎて推論が遅い | 高 | 小型蒸留版を使用（spec記載の0.3秒目標） |
| 合成データと実データのドメインギャップ | 中 | データ拡張強化、Transfer Learning検討 |
| STARメッシュがUnityで正しく表示されない | 中 | SMPL公式FBX/スクリプトを参考 |
| VRM互換性が困難 | 低 | Phase 1ではVRM不要、独自メッシュ使用 |

---

## 次のアクション（First Step）

1. **今日〜明日**: Python環境構築、STAR/Sapiensモデルダウンロード
2. **3日目**: Sapiens推論テスト（test_preprocessing.py）
3. **4-5日目**: 形状推定ネットワーク実装（shape_estimator.py）
4. **6-7日目**: STARレイヤー実装（star_layer.py）
5. **8日目**: 統合テスト（画像→β→メッシュ）

**マイルストーン**: 1週間後に「テスト画像からβパラメータとメッシュ生成」が動作

---

## 参考資料

- [STAR公式サイト](https://star.is.tue.mpg.de/)
- [SMPL Unity実装ガイド](https://files.is.tue.mpg.de/nmahmood/smpl_website/How-to_SMPLinUnity.pdf)
- [Sapiens (Meta AI)](https://github.com/facebookresearch/sapiens)
- [STRAPS論文（BMVC 2020）](https://www.bmvc2020-conference.com/assets/papers/0833.pdf)
- [UniVRM公式ドキュメント](https://vrm.dev/en/univrm/)

---

## 補足: SMPL/STARのUnity実装詳細

### ブレンドシェイプ構造
- **Shape Blendshapes**: βパラメータ（10〜300次元）
  - 各次元が1つのブレンドシェイプに対応
  - 体型変化（身長、体重、プロポーション）を表現

- **Pose Blendshapes**: 414個の補正シェイプ
  - LBS (Linear Blend Skinning) のアーティファクト修正
  - 関節の曲げによる筋肉・脂肪の変形を再現

### ブレンドウェイト変換
```csharp
// SMPL: -1 〜 +1
// Unity: 0 〜 1
float unityWeight = (smplWeight + 1.0f) * 50.0f; // 0〜100スケール
meshRenderer.SetBlendShapeWeight(index, unityWeight);
```

### 推奨設定
- Quality設定: 4 Bones（SMPLは4ボーンスキニング）
- Normals: Import (法線は事前計算済み)
- BlendShapes: Import

---

*作成日: 2025-11-21*
*更新予定: Phase 1完了時に詳細追加*
