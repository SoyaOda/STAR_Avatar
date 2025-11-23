#!/bin/bash

# Sapiens推論 - 単一画像指定モード
# 使用例:
#   ./run_sapiens_single.sh outputs/skin_tone_samples/skin_tone_02_Skin.png
#   ./run_sapiens_single.sh outputs/skin_tone_samples/skin_tone_02_Skin.png outputs/test_output

set -e

# 引数チェック
if [ -z "$1" ]; then
    echo "❌ エラー: 画像パスを指定してください"
    echo ""
    echo "使用例:"
    echo "  $0 <画像パス> [出力ディレクトリ]"
    echo ""
    echo "例:"
    echo "  $0 outputs/skin_tone_samples/skin_tone_02_Skin.png"
    echo "  $0 human_img/human_1.jpg outputs/custom_output"
    exit 1
fi

INPUT_IMAGE="$1"

# 入力画像の存在確認
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "❌ エラー: 画像が見つかりません: $INPUT_IMAGE"
    exit 1
fi

# 絶対パスに変換
INPUT_IMAGE_ABS=$(cd "$(dirname "$INPUT_IMAGE")" && pwd)/$(basename "$INPUT_IMAGE")

# 出力ディレクトリ（引数で指定されていればそれを使用、なければ自動生成）
if [ -n "$2" ]; then
    OUTPUT_BASE="$2"
else
    # 画像ファイル名から出力ディレクトリ名を生成
    IMAGE_NAME=$(basename "$INPUT_IMAGE" | sed 's/\.[^.]*$//')
    OUTPUT_BASE="outputs/sapiens_${IMAGE_NAME}"
fi

OUTPUT_BASE_ABS=$(cd "$(dirname "$0")" && pwd)/$OUTPUT_BASE

# 一時ディレクトリを作成（Sapiensはディレクトリ入力が必要）
TEMP_INPUT_DIR=$(mktemp -d)
trap "rm -rf $TEMP_INPUT_DIR" EXIT

# 画像を一時ディレクトリにコピー
cp "$INPUT_IMAGE_ABS" "$TEMP_INPUT_DIR/"

# Sapiens設定
cd /Users/moei/program/sapiens-main/lite || exit

SAPIENS_CHECKPOINT_ROOT=~/sapiens_lite_host
MODE='torchscript'
SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

echo "========================================================================"
echo "Sapiens推論 - 単一画像モード"
echo "========================================================================"
echo "入力画像: $INPUT_IMAGE_ABS"
echo "出力先:   $OUTPUT_BASE_ABS"
echo ""
echo "モデル構成（最高精度）:"
echo "  - Segmentation: 1B"
echo "  - Normal:       2B ⭐"
echo "  - Depth:        2B ⭐"
echo "  - Pose:         1B"
echo ""

mkdir -p "$OUTPUT_BASE_ABS"

#==============================================================================
# タスク1: セグメンテーション（1B）
#==============================================================================
echo "[1/4] セグメンテーション実行中..."
CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_1b/*.pt2
OUTPUT_SEG="$OUTPUT_BASE_ABS/seg"
mkdir -p "$OUTPUT_SEG"

python3 demo/vis_seg.py \
  $CHECKPOINT \
  --input "$TEMP_INPUT_DIR" \
  --batch-size 1 \
  --output-root="$OUTPUT_SEG" \
  --device cpu 2>&1 | tail -n 3

echo "✓ セグメンテーション完了"
echo ""

#==============================================================================
# タスク2: 法線推定（2B）
#==============================================================================
echo "[2/4] 法線推定実行中（2B - 最高精度）..."
CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_2b/*.pt2
OUTPUT_NORMAL="$OUTPUT_BASE_ABS/normal"
mkdir -p "$OUTPUT_NORMAL"

python3 demo/vis_normal.py \
  $CHECKPOINT \
  --input "$TEMP_INPUT_DIR" \
  --seg_dir "$OUTPUT_SEG" \
  --batch-size 1 \
  --output-root="$OUTPUT_NORMAL" \
  --device cpu 2>&1 | tail -n 3

echo "✓ 法線推定完了"
echo ""

#==============================================================================
# タスク3: 深度推定（2B）
#==============================================================================
echo "[3/4] 深度推定実行中（2B - 最高精度）..."
CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/depth/checkpoints/sapiens_2b/*.pt2
OUTPUT_DEPTH="$OUTPUT_BASE_ABS/depth"
mkdir -p "$OUTPUT_DEPTH"

python3 demo/vis_depth.py \
  $CHECKPOINT \
  --input "$TEMP_INPUT_DIR" \
  --seg_dir "$OUTPUT_SEG" \
  --batch-size 1 \
  --output-root="$OUTPUT_DEPTH" \
  --device cpu 2>&1 | tail -n 3

echo "✓ 深度推定完了"
echo ""

#==============================================================================
# タスク4: 姿勢推定（1B）
#==============================================================================
echo "[4/4] 姿勢推定実行中..."
CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/*.pt2
OUTPUT_POSE="$OUTPUT_BASE_ABS/pose"
mkdir -p "$OUTPUT_POSE"

python3 demo/vis_pose.py \
  $CHECKPOINT \
  --input "$TEMP_INPUT_DIR" \
  --batch-size 1 \
  --output-root="$OUTPUT_POSE" \
  --device cpu 2>&1 | tail -n 3

echo "✓ 姿勢推定完了"
echo ""

#==============================================================================
# 結果の確認
#==============================================================================
echo "========================================================================"
echo "✅ Sapiens推論完了！"
echo "========================================================================"
echo ""
echo "入力画像: $(basename "$INPUT_IMAGE_ABS")"
echo ""
echo "生成されたファイル:"
echo ""
echo "📁 セグメンテーション ($OUTPUT_SEG):"
ls -lh "$OUTPUT_SEG"/*.jpg 2>/dev/null | awk '{print "  ", $9, "("$5")"}' || echo "  (ファイルなし)"
echo ""

echo "📁 法線推定 - 2B ($OUTPUT_NORMAL):"
ls -lh "$OUTPUT_NORMAL"/*.jpg 2>/dev/null | awk '{print "  ", $9, "("$5")"}' || echo "  (ファイルなし)"
echo ""

echo "📁 深度推定 - 2B ($OUTPUT_DEPTH):"
ls -lh "$OUTPUT_DEPTH"/*.jpg 2>/dev/null | awk '{print "  ", $9, "("$5")"}' || echo "  (ファイルなし)"
echo ""

echo "📁 姿勢推定 ($OUTPUT_POSE):"
ls -lh "$OUTPUT_POSE"/*.json 2>/dev/null | awk '{print "  ", $9, "("$5")"}' || echo "  (ファイルなし)"
echo ""

echo "========================================================================"
echo "結果を確認: open $OUTPUT_BASE_ABS"
echo "========================================================================"
