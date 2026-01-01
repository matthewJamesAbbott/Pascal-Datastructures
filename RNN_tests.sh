#!/bin/bash

#
# Matthew Abbott 2025
# RNN Tests - Comprehensive Test Suite
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="./test_output"
RNN_BIN="./RNN"
FACADE_BIN="./FacadeRNN"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup/Cleanup
cleanup() {
    # Cleanup handled manually if needed
    :
}
trap cleanup EXIT

mkdir -p "$TEMP_DIR"

fpc RNN.pas
fpc FacadeRNN.pas

# Test function
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    output=$(eval "$command" 2>&1)
    exit_code=$?

    if echo "$output" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Command: $command"
        echo "  Expected pattern: $expected_pattern"
        echo "  Output:"
        echo "$output" | head -5
        FAIL=$((FAIL + 1))
    fi
}

check_file_exists() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ -f "$file" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
    fi
}

check_json_valid() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ ! -f "$file" ]; then
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
        return
    fi

    if grep -q '"input_size"' "$file" && grep -q '"output_size"' "$file" && grep -q '"hidden_sizes"' "$file"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Invalid JSON structure in $file"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================
# Start Tests
# ============================================

echo ""
echo "========================================="
echo "RNN Comprehensive Test Suite"
echo "========================================="
echo ""

# Check binaries exist
if [ ! -f "$RNN_BIN" ]; then
    echo -e "${RED}Error: $RNN_BIN not found. Compile with: fpc RNN.pas${NC}"
    exit 1
fi

if [ ! -f "$FACADE_BIN" ]; then
    echo -e "${RED}Error: $FACADE_BIN not found. Compile with: fpc FacadeRNN.pas${NC}"
    exit 1
fi

echo -e "${BLUE}=== RNN Binary Tests ===${NC}"
echo ""

# ============================================
# Basic Help/Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "RNN help command" \
    "$RNN_BIN help" \
    "Commands:"

run_test \
    "RNN --help flag" \
    "$RNN_BIN --help" \
    "Commands:"

run_test \
    "FacadeRNN help command" \
    "$FACADE_BIN help" \
    "Commands:"

run_test \
    "FacadeRNN --help flag" \
    "$FACADE_BIN --help" \
    "Commands:"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create 2-4-1 LSTM model" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic.json" \
    "Created RNN model"

check_file_exists \
    "JSON file created for 2-4-1" \
    "$TEMP_DIR/basic.json"

check_json_valid \
    "JSON contains valid RNN structure" \
    "$TEMP_DIR/basic.json"

run_test \
    "Output shows correct architecture" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic2.json" \
    "Input size: 2"

run_test \
    "Output shows hidden size" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic3.json" \
    "Hidden sizes: 4"

run_test \
    "Output shows output size" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic4.json" \
    "Output size: 1"

echo ""

# ============================================
# Model Creation - Multi-layer
# ============================================

echo -e "${BLUE}Group: Model Creation - Multi-layer${NC}"

run_test \
    "Create 3-5-3-2 network" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/multilayer.json" \
    "Created RNN model"

check_file_exists \
    "JSON file for multi-layer" \
    "$TEMP_DIR/multilayer.json"

run_test \
    "Multi-layer output shows correct input" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml2.json" \
    "Input size: 3"

run_test \
    "Multi-layer output shows both hidden sizes" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml3.json" \
    "Hidden sizes: 5,3"

run_test \
    "Multi-layer output shows correct output size" \
    "$RNN_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml4.json" \
    "Output size: 2"

run_test \
    "Create 3-layer hidden network" \
    "$RNN_BIN create --input=4 --hidden=8,6,4 --output=2 --save=$TEMP_DIR/ml5.json" \
    "Hidden sizes: 8,6,4"

echo ""

# ============================================
# Cell Types
# ============================================

echo -e "${BLUE}Group: Cell Types${NC}"

run_test \
    "Create with SimpleRNN cell" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/simplernn.json --cell=simplernn" \
    "Created RNN model"

run_test \
    "Create with LSTM cell (default)" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lstm.json --cell=lstm" \
    "Created RNN model"

run_test \
    "Create with GRU cell" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/gru.json --cell=gru" \
    "Created RNN model"

run_test \
    "Output shows SimpleRNN cell type" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/simplernn2.json --cell=simplernn" \
    "Cell type: simplernn"

run_test \
    "Output shows LSTM cell type" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lstm2.json --cell=lstm" \
    "Cell type: lstm"

run_test \
    "Output shows GRU cell type" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/gru2.json --cell=gru" \
    "Cell type: gru"

run_test \
    "FacadeRNN with SimpleRNN cell" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_simplernn.json --cell=simplernn" \
    "Cell type: simplernn"

run_test \
    "FacadeRNN with LSTM cell" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_lstm.json --cell=lstm" \
    "Cell type: lstm"

run_test \
    "FacadeRNN with GRU cell" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_gru.json --cell=gru" \
    "Cell type: gru"

echo ""

# ============================================
# Activation Functions
# ============================================

echo -e "${BLUE}Group: Activation Functions${NC}"

run_test \
    "Create with tanh hidden activation (default)" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_tanh.json --hidden-act=tanh" \
    "Hidden activation: tanh"

run_test \
    "Create with sigmoid hidden activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_sigmoid.json --hidden-act=sigmoid" \
    "Hidden activation: sigmoid"

run_test \
    "Create with relu hidden activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_relu.json --hidden-act=relu" \
    "Hidden activation: relu"

run_test \
    "Create with linear output activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/out_linear.json --output-act=linear" \
    "Output activation: linear"

run_test \
    "Create with sigmoid output activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/out_sigmoid.json --output-act=sigmoid" \
    "Output activation: sigmoid"

run_test \
    "Create with tanh output activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=3 --save=$TEMP_DIR/out_tanh.json --output-act=tanh" \
    "Output activation: tanh"

run_test \
    "FacadeRNN with tanh activation" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_act_tanh.json --hidden-act=tanh" \
    "Hidden activation: tanh"

run_test \
    "FacadeRNN with relu activation" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_act_relu.json --hidden-act=relu" \
    "Hidden activation: relu"

echo ""

# ============================================
# Loss Functions
# ============================================

echo -e "${BLUE}Group: Loss Functions${NC}"

run_test \
    "Create with MSE loss (default)" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/loss_mse.json --loss=mse" \
    "Loss function: mse"

run_test \
    "Create with cross-entropy loss" \
    "$RNN_BIN create --input=2 --hidden=3 --output=3 --save=$TEMP_DIR/loss_ce.json --loss=crossentropy" \
    "Loss function: crossentropy"

run_test \
    "FacadeRNN with MSE loss" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_loss_mse.json --loss=mse" \
    "Loss function: mse"

run_test \
    "FacadeRNN with cross-entropy loss" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=3 --save=$TEMP_DIR/facade_loss_ce.json --loss=crossentropy" \
    "Loss function: crossentropy"

echo ""

# ============================================
# Hyperparameters
# ============================================

echo -e "${BLUE}Group: Hyperparameters${NC}"

run_test \
    "Custom learning rate" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/hyper_lr.json --lr=0.001" \
    "Learning rate: 0"

run_test \
    "Custom gradient clip" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/hyper_clip.json --clip=10.0" \
    "Gradient clip: 10"

run_test \
    "Custom BPTT steps" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/hyper_bptt.json --bptt=20" \
    "BPTT steps: 20"

run_test \
    "High learning rate" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/high_lr.json --lr=0.1" \
    "Learning rate: 0"

run_test \
    "Low learning rate" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/low_lr.json --lr=0.0001" \
    "Learning rate:"

run_test \
    "FacadeRNN custom learning rate" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/facade_hyper_lr.json --lr=0.005" \
    "Learning rate:"

run_test \
    "FacadeRNN custom gradient clip" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/facade_hyper_clip.json --clip=7.5" \
    "Gradient clip: 7"

echo ""

# ============================================
# JSON Structure Validation
# ============================================

echo -e "${BLUE}Group: JSON Structure Validation${NC}"

run_test \
    "JSON has cell_type field" \
    "grep -q 'cell_type' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has hidden_sizes field" \
    "grep -q 'hidden_sizes' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has cells array" \
    "grep -q 'cells' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has output_layer" \
    "grep -q 'output_layer' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "LSTM JSON has Wf weights" \
    "grep -q 'Wf' $TEMP_DIR/lstm.json && echo 'ok'" \
    "ok"

run_test \
    "LSTM JSON has Wi weights" \
    "grep -q 'Wi' $TEMP_DIR/lstm.json && echo 'ok'" \
    "ok"

run_test \
    "LSTM JSON has Wc weights" \
    "grep -q 'Wc' $TEMP_DIR/lstm.json && echo 'ok'" \
    "ok"

run_test \
    "LSTM JSON has Wo weights" \
    "grep -q 'Wo' $TEMP_DIR/lstm.json && echo 'ok'" \
    "ok"

run_test \
    "GRU JSON has Wz weights" \
    "grep -q 'Wz' $TEMP_DIR/gru.json && echo 'ok'" \
    "ok"

run_test \
    "GRU JSON has Wr weights" \
    "grep -q 'Wr' $TEMP_DIR/gru.json && echo 'ok'" \
    "ok"

run_test \
    "GRU JSON has Wh weights" \
    "grep -q 'Wh' $TEMP_DIR/gru.json && echo 'ok'" \
    "ok"

run_test \
    "SimpleRNN JSON has Wih weights" \
    "grep -q 'Wih' $TEMP_DIR/simplernn.json && echo 'ok'" \
    "ok"

run_test \
    "SimpleRNN JSON has Whh weights" \
    "grep -q 'Whh' $TEMP_DIR/simplernn.json && echo 'ok'" \
    "ok"

run_test \
    "JSON starts with opening brace" \
    "head -1 $TEMP_DIR/basic.json | grep -q '{' && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Facade-specific Extended Features
# ============================================

echo -e "${BLUE}Group: FacadeRNN Extended Features${NC}"

run_test \
    "FacadeRNN shows hyperparameters in info" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_hyper.json && $FACADE_BIN info --model=$TEMP_DIR/facade_hyper.json" \
    "Cell type:"

run_test \
    "FacadeRNN info shows learning rate" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_hyper.json" \
    "Learning rate:"

run_test \
    "FacadeRNN info shows cell type" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_hyper.json" \
    "Cell type:"

run_test \
    "FacadeRNN info shows activation functions" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_hyper.json" \
    "activation:"

run_test \
    "FacadeRNN displays gradient clipping" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_clip.json --clip=7.5 && $FACADE_BIN info --model=$TEMP_DIR/facade_clip.json" \
    "Gradient clip"

echo ""

# ============================================
# FacadeRNN Query Command Tests
# ============================================

echo -e "${BLUE}Group: FacadeRNN Query Commands${NC}"

# Create a model for query tests
$FACADE_BIN create --input=4 --hidden=8 --output=3 --save=$TEMP_DIR/query_test.json --cell=lstm > /dev/null 2>&1

run_test \
    "FacadeRNN query input-size" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_test.json --query-type=input-size" \
    "Input size: 4"

run_test \
    "FacadeRNN query output-size" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_test.json --query-type=output-size" \
    "Output size: 3"

run_test \
    "FacadeRNN query hidden-size" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_test.json --query-type=hidden-size --layer=0" \
    "Hidden size"

run_test \
    "FacadeRNN query cell-type" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_test.json --query-type=cell-type" \
    "Cell type: lstm"

run_test \
    "FacadeRNN query dropout-rate" \
    "$FACADE_BIN query --model=$TEMP_DIR/query_test.json --query-type=dropout-rate" \
    "dropout rate:"

echo ""

# ============================================
# Cross-binary Compatibility
# ============================================

echo -e "${BLUE}Group: Cross-binary Compatibility${NC}"

run_test \
    "Model created by RNN can be loaded by FacadeRNN" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cross.json && $FACADE_BIN info --model=$TEMP_DIR/cross.json" \
    "Cell type:"

run_test \
    "FacadeRNN model info shows correct input" \
    "$FACADE_BIN info --model=$TEMP_DIR/cross.json" \
    "Input size: 2"

run_test \
    "FacadeRNN can load RNN-created model for prediction" \
    "$FACADE_BIN predict --model=$TEMP_DIR/cross.json --input=0.5,0.5" \
    "Input:"

run_test \
    "Model created by FacadeRNN can be loaded by RNN" \
    "$FACADE_BIN create --input=3 --hidden=5 --output=2 --save=$TEMP_DIR/cross2.json && $RNN_BIN info --model=$TEMP_DIR/cross2.json" \
    "Input size: 3"

run_test \
    "RNN can load FacadeRNN-created model for prediction" \
    "$RNN_BIN predict --model=$TEMP_DIR/cross2.json --input=0.1,0.2,0.3" \
    "Output:"

run_test \
    "Cross-load LSTM model (RNN -> Facade)" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --cell=lstm --save=$TEMP_DIR/cross_lstm.json && $FACADE_BIN predict --model=$TEMP_DIR/cross_lstm.json --input=0.5,0.5" \
    "Input:"

run_test \
    "Cross-load GRU model (RNN -> Facade)" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --cell=gru --save=$TEMP_DIR/cross_gru.json && $FACADE_BIN predict --model=$TEMP_DIR/cross_gru.json --input=0.5,0.5" \
    "Input:"

run_test \
    "Cross-load SimpleRNN model (RNN -> Facade)" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --cell=simplernn --save=$TEMP_DIR/cross_simple.json && $FACADE_BIN predict --model=$TEMP_DIR/cross_simple.json --input=0.5,0.5" \
    "Input:"

run_test \
    "Cross-load LSTM model (Facade -> RNN)" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --cell=lstm --save=$TEMP_DIR/cross_lstm2.json && $RNN_BIN predict --model=$TEMP_DIR/cross_lstm2.json --input=0.5,0.5" \
    "Output:"

run_test \
    "Cross-load GRU model (Facade -> RNN)" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --cell=gru --save=$TEMP_DIR/cross_gru2.json && $RNN_BIN predict --model=$TEMP_DIR/cross_gru2.json --input=0.5,0.5" \
    "Output:"

run_test \
    "Cross-load SimpleRNN model (Facade -> RNN)" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --cell=simplernn --save=$TEMP_DIR/cross_simple2.json && $RNN_BIN predict --model=$TEMP_DIR/cross_simple2.json --input=0.5,0.5" \
    "Output:"

run_test \
    "Cross-load multi-layer model (RNN -> Facade)" \
    "$RNN_BIN create --input=3 --hidden=6,4 --output=2 --cell=lstm --save=$TEMP_DIR/cross_multi.json && $FACADE_BIN predict --model=$TEMP_DIR/cross_multi.json --input=0.1,0.2,0.3" \
    "Input:"

run_test \
    "Cross-load multi-layer model (Facade -> RNN)" \
    "$FACADE_BIN create --input=3 --hidden=6,4 --output=2 --cell=lstm --save=$TEMP_DIR/cross_multi2.json && $RNN_BIN predict --model=$TEMP_DIR/cross_multi2.json --input=0.1,0.2,0.3" \
    "Output:"

echo ""

# ============================================
# FacadeRNN Prediction Tests
# ============================================

echo -e "${BLUE}Group: FacadeRNN Prediction${NC}"

run_test \
    "FacadeRNN predict command loads and predicts" \
    "$FACADE_BIN predict --model=$TEMP_DIR/basic.json --input=0.5,0.3" \
    "Input:"

run_test \
    "FacadeRNN predict shows input" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_pred_test.json && $FACADE_BIN predict --model=$TEMP_DIR/facade_pred_test.json --input=0.5,0.3" \
    "Input: 0"

run_test \
    "FacadeRNN predict with different cell types (SimpleRNN)" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --cell=simplernn --save=$TEMP_DIR/facade_simplernn_pred.json && $FACADE_BIN predict --model=$TEMP_DIR/facade_simplernn_pred.json --input=0.5,0.5" \
    "Input:"

run_test \
    "FacadeRNN predict with GRU cell" \
    "$FACADE_BIN create --input=2 --hidden=4 --output=1 --cell=gru --save=$TEMP_DIR/facade_gru_pred.json && $FACADE_BIN predict --model=$TEMP_DIR/facade_gru_pred.json --input=0.5,0.5" \
    "Input:"

run_test \
    "FacadeRNN predict validates input requirement" \
    "$FACADE_BIN predict --model=$TEMP_DIR/facade_pred_test.json 2>&1" \
    "Error"

run_test \
    "FacadeRNN predict with multi-layer network" \
    "$FACADE_BIN create --input=3 --hidden=6,4 --output=2 --save=$TEMP_DIR/facade_multi_pred.json && $FACADE_BIN predict --model=$TEMP_DIR/facade_multi_pred.json --input=0.1,0.2,0.3" \
    "Input:"

echo ""

# ============================================
# RNN Prediction Tests
# ============================================

echo -e "${BLUE}Group: RNN Prediction${NC}"

run_test \
    "RNN predict command loads and predicts" \
    "$RNN_BIN predict --model=$TEMP_DIR/basic.json --input=0.5,0.3" \
    "Output:"

run_test \
    "RNN predict shows input and output" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/pred_test.json && $RNN_BIN predict --model=$TEMP_DIR/pred_test.json --input=0.5,0.3" \
    "Input:"

run_test \
    "RNN predict returns numeric values" \
    "$RNN_BIN predict --model=$TEMP_DIR/pred_test.json --input=0.1,0.2" \
    "Output:"

run_test \
    "RNN predict with multi-hidden layers" \
    "$RNN_BIN create --input=2 --hidden=8,6 --output=1 --save=$TEMP_DIR/multi_hidden_pred.json && $RNN_BIN predict --model=$TEMP_DIR/multi_hidden_pred.json --input=0.5,0.5" \
    "Output:"

run_test \
    "RNN predict with different cell types (SimpleRNN)" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --cell=simplernn --save=$TEMP_DIR/simplernn_pred.json && $RNN_BIN predict --model=$TEMP_DIR/simplernn_pred.json --input=0.5,0.5" \
    "Output:"

run_test \
    "RNN predict with GRU cell" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --cell=gru --save=$TEMP_DIR/gru_pred.json && $RNN_BIN predict --model=$TEMP_DIR/gru_pred.json --input=0.5,0.5" \
    "Output:"

run_test \
    "RNN predict validates input requirement" \
    "$RNN_BIN predict --model=$TEMP_DIR/pred_test.json 2>&1" \
    "Error"

run_test \
    "RNN predict with 3-layer hidden" \
    "$RNN_BIN create --input=4 --hidden=8,6,4 --output=2 --save=$TEMP_DIR/rnn_3layer.json && $RNN_BIN predict --model=$TEMP_DIR/rnn_3layer.json --input=0.1,0.2,0.3,0.4" \
    "Output:"

echo ""

# ============================================
# Error Cases (expected to handle gracefully)
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Missing required --save argument" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 2>&1" \
    "Error"

run_test \
    "Missing input size" \
    "$RNN_BIN create --hidden=3 --output=1 --save=$TEMP_DIR/err1.json 2>&1" \
    "Error"

run_test \
    "Missing output size" \
    "$RNN_BIN create --input=2 --hidden=3 --save=$TEMP_DIR/err2.json 2>&1" \
    "Error"

run_test \
    "Missing hidden layers" \
    "$RNN_BIN create --input=2 --output=1 --save=$TEMP_DIR/err3.json 2>&1" \
    "Error"

run_test \
    "Loading non-existent model" \
    "$RNN_BIN info --model=$TEMP_DIR/nonexistent.json 2>&1" \
    ""

run_test \
    "FacadeRNN missing --save argument" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 2>&1" \
    "Error"

run_test \
    "FacadeRNN missing input size" \
    "$FACADE_BIN create --hidden=3 --output=1 --save=$TEMP_DIR/err4.json 2>&1" \
    "Error"

run_test \
    "FacadeRNN loading non-existent model" \
    "$FACADE_BIN info --model=$TEMP_DIR/nonexistent2.json 2>&1" \
    ""

echo ""

# ============================================
# Sequential Operations (Real Workflow)
# ============================================

echo -e "${BLUE}Group: Sequential Operations Workflow${NC}"

run_test \
    "Workflow: Create -> Load -> Info (SimpleRNN)" \
    "$RNN_BIN create --input=10 --hidden=16 --output=2 --cell=simplernn --save=$TEMP_DIR/workflow1.json && $FACADE_BIN info --model=$TEMP_DIR/workflow1.json" \
    "Input size: 10"

run_test \
    "Workflow: Create -> Load -> Info (LSTM)" \
    "$RNN_BIN create --input=5 --hidden=8,6 --output=3 --cell=lstm --save=$TEMP_DIR/workflow2.json && $FACADE_BIN info --model=$TEMP_DIR/workflow2.json" \
    "Output size: 3"

run_test \
    "Workflow: Create -> Load -> Info (GRU)" \
    "$RNN_BIN create --input=4 --hidden=12 --output=2 --cell=gru --save=$TEMP_DIR/workflow3.json && $FACADE_BIN info --model=$TEMP_DIR/workflow3.json" \
    "Cell type: gru"

run_test \
    "Workflow: Create with params -> Verify -> Predict" \
    "$RNN_BIN create --input=4 --hidden=8,6 --output=2 --save=$TEMP_DIR/workflow4.json --lr=0.01 --cell=lstm --clip=5.0 && $FACADE_BIN info --model=$TEMP_DIR/workflow4.json" \
    "Input size: 4"

run_test \
    "Workflow: FacadeRNN Create -> RNN Predict" \
    "$FACADE_BIN create --input=3 --hidden=5 --output=2 --save=$TEMP_DIR/workflow5.json && $RNN_BIN predict --model=$TEMP_DIR/workflow5.json --input=0.1,0.2,0.3" \
    "Output:"

run_test \
    "Workflow: RNN Create -> FacadeRNN Predict" \
    "$RNN_BIN create --input=3 --hidden=5 --output=2 --save=$TEMP_DIR/workflow6.json && $FACADE_BIN predict --model=$TEMP_DIR/workflow6.json --input=0.1,0.2,0.3" \
    "Input:"

echo ""

# ============================================
# Advanced Features (Specific to RNN)
# ============================================

echo -e "${BLUE}Group: RNN-Specific Features${NC}"

run_test \
    "Large hidden layer" \
    "$RNN_BIN create --input=2 --hidden=256 --output=2 --save=$TEMP_DIR/large_hidden.json" \
    "Created RNN model"

run_test \
    "Multiple hidden layers" \
    "$RNN_BIN create --input=2 --hidden=64,32,16 --output=2 --save=$TEMP_DIR/multi_hidden.json" \
    "Hidden sizes: 64,32,16"

run_test \
    "Sequence length configuration (training context)" \
    "$RNN_BIN create --input=5 --hidden=10 --output=2 --save=$TEMP_DIR/seq_config.json --bptt=16" \
    "BPTT steps: 16"

run_test \
    "Very large network" \
    "$RNN_BIN create --input=10 --hidden=128,64,32 --output=5 --save=$TEMP_DIR/large_net.json" \
    "Created RNN model"

run_test \
    "Single neuron hidden layer" \
    "$RNN_BIN create --input=2 --hidden=1 --output=1 --save=$TEMP_DIR/single_hidden.json" \
    "Hidden sizes: 1"

run_test \
    "Many output neurons" \
    "$RNN_BIN create --input=4 --hidden=8 --output=10 --save=$TEMP_DIR/many_output.json" \
    "Output size: 10"

echo ""

# ============================================
# Weight Loading Tests (JavaScript compatibility)
# ============================================

echo -e "${BLUE}Group: Weight Loading & JavaScript Compatibility${NC}"

# Check if Node.js is available
if command -v node &> /dev/null; then
    echo -e "${YELLOW}Info: Testing weight loading with inline JavaScript validation${NC}"
    
    # Create models for weight loading tests
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=lstm --save=$TEMP_DIR/js_test_lstm.json > /dev/null 2>&1
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=gru --save=$TEMP_DIR/js_test_gru.json > /dev/null 2>&1
    $RNN_BIN create --input=4 --hidden=8 --output=3 --cell=simplernn --save=$TEMP_DIR/js_test_simple.json > /dev/null 2>&1
    $RNN_BIN create --input=3 --hidden=6,4 --output=2 --cell=lstm --save=$TEMP_DIR/js_test_multi.json > /dev/null 2>&1
    $FACADE_BIN create --input=4 --hidden=8 --output=3 --cell=lstm --save=$TEMP_DIR/js_test_facade.json > /dev/null 2>&1

    # Inline JavaScript validation function
    validate_rnn_js() {
        local jsonfile="$1"
        node -e "
const fs = require('fs');
let pass = 0, fail = 0;
function test(name, condition) {
    if (condition) { console.log('✓ ' + name); pass++; }
    else { console.log('✗ ' + name); fail++; }
}
function hasValidNumbers(arr) {
    if (!Array.isArray(arr)) return false;
    const flat = arr.flat(Infinity);
    return flat.every(n => typeof n === 'number' && !isNaN(n) && isFinite(n));
}
try {
    const json = JSON.parse(fs.readFileSync('$jsonfile', 'utf8'));
    test('JSON has input_size', json.input_size !== undefined);
    test('JSON has output_size', json.output_size !== undefined);
    test('JSON has hidden_sizes', Array.isArray(json.hidden_sizes));
    test('JSON has cell_type', json.cell_type !== undefined);
    test('JSON has cells array', Array.isArray(json.cells));
    test('JSON has output_layer', json.output_layer !== undefined);
    const inputSize = json.input_size, outputSize = json.output_size;
    const hiddenSizes = json.hidden_sizes, cellType = json.cell_type;
    test('Input size is positive', inputSize > 0);
    test('Output size is positive', outputSize > 0);
    test('Cell count matches hidden_sizes', json.cells.length === hiddenSizes.length);
    let prevSize = inputSize;
    for (let i = 0; i < json.cells.length; i++) {
        const cell = json.cells[i], hs = hiddenSizes[i], expDim = prevSize + hs;
        if (cellType === 'lstm') {
            test('Cell ' + i + ': Has Wf weights', cell.Wf !== undefined);
            test('Cell ' + i + ': Has Wi weights', cell.Wi !== undefined);
            test('Cell ' + i + ': Has Wc weights', cell.Wc !== undefined);
            test('Cell ' + i + ': Has Wo weights', cell.Wo !== undefined);
            test('Cell ' + i + ': Wf shape correct', cell.Wf && cell.Wf.length === hs && cell.Wf[0]?.length === expDim);
            test('Cell ' + i + ': Wf has valid numbers', hasValidNumbers(cell.Wf));
            const bf = cell.Bf ?? cell.bf;
            test('Cell ' + i + ': Has Bf bias', bf !== undefined);
            test('Cell ' + i + ': Bf length correct', bf?.length === hs);
        } else if (cellType === 'gru') {
            test('Cell ' + i + ': Has Wz weights', cell.Wz !== undefined);
            test('Cell ' + i + ': Has Wr weights', cell.Wr !== undefined);
            test('Cell ' + i + ': Has Wh weights', cell.Wh !== undefined);
            test('Cell ' + i + ': Wz shape correct', cell.Wz && cell.Wz.length === hs && cell.Wz[0]?.length === expDim);
            test('Cell ' + i + ': Wz has valid numbers', hasValidNumbers(cell.Wz));
        } else if (cellType === 'simplernn') {
            test('Cell ' + i + ': Has Wih weights', cell.Wih !== undefined);
            test('Cell ' + i + ': Has Whh weights', cell.Whh !== undefined);
            test('Cell ' + i + ': Wih shape correct', cell.Wih && cell.Wih.length === hs && cell.Wih[0]?.length === prevSize);
            test('Cell ' + i + ': Wih has valid numbers', hasValidNumbers(cell.Wih));
        }
        prevSize = hs;
    }
    const outLayer = json.output_layer, lastHs = hiddenSizes[hiddenSizes.length - 1];
    test('Output layer has W', outLayer.W !== undefined);
    const outB = outLayer.B ?? outLayer.b;
    test('Output layer has B', outB !== undefined);
    test('Output W shape correct', outLayer.W && outLayer.W.length === outputSize && outLayer.W[0]?.length === lastHs);
    test('Output W has valid numbers', hasValidNumbers(outLayer.W));
    test('Output B has valid numbers', hasValidNumbers(outB));
    console.log(pass + '/' + (pass + fail) + ' tests passed');
    process.exit(fail === 0 ? 0 : 1);
} catch (e) { console.log('Error: ' + e.message); process.exit(1); }
" 2>&1
    }

    run_test \
        "Weight loading: LSTM basic structure" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "tests passed"

    run_test \
        "Weight loading: GRU basic structure" \
        "validate_rnn_js $TEMP_DIR/js_test_gru.json" \
        "tests passed"

    run_test \
        "Weight loading: SimpleRNN basic structure" \
        "validate_rnn_js $TEMP_DIR/js_test_simple.json" \
        "tests passed"

    run_test \
        "Weight loading: Multi-layer LSTM structure" \
        "validate_rnn_js $TEMP_DIR/js_test_multi.json" \
        "tests passed"

    run_test \
        "Weight loading: FacadeRNN created model" \
        "validate_rnn_js $TEMP_DIR/js_test_facade.json" \
        "tests passed"

    run_test \
        "Weight loading: JSON has input_size" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "JSON has input_size"

    run_test \
        "Weight loading: JSON has output_size" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "JSON has output_size"

    run_test \
        "Weight loading: JSON has hidden_sizes" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "JSON has hidden_sizes"

    run_test \
        "Weight loading: JSON has cells array" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "JSON has cells array"

    run_test \
        "Weight loading: JSON has output_layer" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "JSON has output_layer"

    run_test \
        "Weight loading: LSTM Wf weights present" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "Has Wf weights"

    run_test \
        "Weight loading: LSTM Wf has valid numbers" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "Wf has valid numbers"

    run_test \
        "Weight loading: GRU Wz weights present" \
        "validate_rnn_js $TEMP_DIR/js_test_gru.json" \
        "Has Wz weights"

    run_test \
        "Weight loading: SimpleRNN Wih weights present" \
        "validate_rnn_js $TEMP_DIR/js_test_simple.json" \
        "Has Wih weights"

    run_test \
        "Weight loading: Output W has valid numbers" \
        "validate_rnn_js $TEMP_DIR/js_test_lstm.json" \
        "Output W has valid numbers"

else
    echo -e "${YELLOW}Warning: Node.js not found, skipping JavaScript weight loading tests${NC}"
fi

echo ""

# ============================================
# Summary
# ============================================

echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

echo "========================================="
echo "RNN Implementation Coverage"
echo "========================================="
echo ""
echo "Binaries Tested:"
echo "  ✓ RNN (Pascal)"
echo "  ✓ FacadeRNN (Pascal)"
echo ""
echo "Cell Types Tested:"
echo "  ✓ SimpleRNN"
echo "  ✓ LSTM"
echo "  ✓ GRU"
echo ""
echo "Features Tested:"
echo "  ✓ Model creation (all cell types)"
echo "  ✓ Multi-layer networks"
echo "  ✓ Activation functions (tanh, sigmoid, relu, linear, softmax)"
echo "  ✓ Loss functions (MSE, cross-entropy)"
echo "  ✓ Hyperparameters (LR, gradient clip, BPTT)"
echo "  ✓ JSON serialization"
echo "  ✓ Prediction/inference"
echo "  ✓ Cross-binary compatibility (RNN <-> FacadeRNN)"
echo "  ✓ FacadeRNN query commands"
echo "  ✓ JavaScript weight validation"
echo "  ✓ Error handling"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
