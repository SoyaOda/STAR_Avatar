#!/bin/bash

# Sapiens推論スクリプト（修正版）
# 正しい順序：1. Seg → 2. Seg結果を使ってNormal/Depth

set -e

cd /Users/moei/program/sapiens-main/lite || exit

SAPIENS_CHECKPOINT_ROOT=~/sapiens_lite_host
MODE='torchscript'
SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

# 入力・出力ディレクトリ
INPUT_DIR="/Users/moei/program/STAR_Avatar/human_img"
OUTPUT_BASE="/Users/moei/program/STAR_Avatar/outputs/sapiens_fixed"

MODEL_NAME='sapiens_0.3b'

echo "========================================================================"
echo "Sapiens推論（修正版）- 正しい順序で実行"
echo "========================================================================"
echo "入力: $INPUT_DIR"
echo "出力: $OUTPUT_BASE"
echo "モデル: $MODEL_NAME"
echo ""

# 出力ディレクトリを作成
mkdir -p "$OUTPUT_BASE"

#==============================================================================
# ステップ1: セグメンテーション（最初に実行が必須）
#==============================================================================
echo "ステップ 1/4: セグメンテーション実行中..."
CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/$MODEL_NAME/*.pt2
OUTPUT_SEG="$OUTPUT_BASE/seg"
mkdir -p "$OUTPUT_SEG"

python3 demo/vis_seg.py \
  $CHECKPOINT \
  --input "$INPUT_DIR" \
  --batch-size 1 \
  --output-root="$OUTPUT_SEG" \
  --device cpu

echo "✓ セグメンテーション完了"
echo ""

#==============================================================================
# ステップ2: 法線推定（セグメンテーション結果を使用）
#==============================================================================
echo "ステップ 2/4: 法線推定実行中（セグメンテーションマスク使用）..."
CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/$MODEL_NAME/*.pt2
OUTPUT_NORMAL="$OUTPUT_BASE/normal"
mkdir -p "$OUTPUT_NORMAL"

python3 demo/vis_normal.py \
  $CHECKPOINT \
  --input "$INPUT_DIR" \
  --seg_dir "$OUTPUT_SEG" \
  --batch-size 1 \
  --output-root="$OUTPUT_NORMAL" \
  --device cpu

echo "✓ 法線推定完了"
echo ""

#==============================================================================
# ステップ3: 深度推定（セグメンテーション結果を使用）
#==============================================================================
echo "ステップ 3/4: 深度推定実行中（セグメンテーションマスク使用）..."
CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/depth/checkpoints/$MODEL_NAME/*.pt2
OUTPUT_DEPTH="$OUTPUT_BASE/depth"
mkdir -p "$OUTPUT_DEPTH"

python3 demo/vis_depth.py \
  $CHECKPOINT \
  --input "$INPUT_DIR" \
  --seg_dir "$OUTPUT_SEG" \
  --batch-size 1 \
  --output-root="$OUTPUT_DEPTH" \
  --device cpu

echo "✓ 深度推定完了"
echo ""

#==============================================================================
# ステップ4: 姿勢推定
#==============================================================================
echo "ステップ 4/4: 姿勢推定実行中..."
CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/$MODEL_NAME/*.pt2
OUTPUT_POSE="$OUTPUT_BASE/pose"
mkdir -p "$OUTPUT_POSE"

python3 demo/vis_pose.py \
  $CHECKPOINT \
  --input "$INPUT_DIR" \
  --batch-size 1 \
  --output-root="$OUTPUT_POSE" \
  --device cpu

echo "✓ 姿勢推定完了"
echo ""

#==============================================================================
# 完了
#==============================================================================
echo "========================================================================"
echo "✓ すべての処理が完了しました！"
echo "========================================================================"
echo ""
echo "結果:"
echo "  セグメンテーション: $OUTPUT_SEG"
echo "  法線推定:          $OUTPUT_NORMAL"
echo "  深度推定:          $OUTPUT_DEPTH"
echo "  姿勢推定:          $OUTPUT_POSE"
echo ""
echo "各ディレクトリには以下が含まれます:"
echo "  - .jpg または .png: 可視化画像（入力+結果の並置）"
echo "  - .npy: 生データ（プログラムで使用可能）"
echo ""
echo "結果を確認: open $OUTPUT_BASE"
echo "========================================================================"


cd /Users/moei/program/sapiens-main && python3 -c "from download_models import download_model; 
download_model('depth', 'sapiens_2b')"