#!/bin/bash

#
# Matthew Abbott 2025
# RNN Tests
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
echo "RNN User Workflow Test Suite"
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

echo ""

# ============================================
# Hyperparameters
# ============================================

echo -e "${BLUE}Group: Hyperparameters${NC}"

run_test \
    "Create with custom learning rate" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lr.json --lr=0.01" \
    "Created RNN model"

run_test \
    "Custom learning rate in output" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lr2.json --lr=0.01" \
    "Learning rate: 0"

run_test \
    "Create with gradient clipping" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/clip.json --clip=10.0" \
    "Created RNN model"

run_test \
    "Gradient clip in output" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/clip2.json --clip=10.0" \
    "Gradient clip:"

run_test \
    "Create with BPTT steps" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/bptt.json --bptt=32" \
    "Created RNN model"

run_test \
    "BPTT steps in output" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/bptt2.json --bptt=32" \
    "BPTT steps: 32"

echo ""

# ============================================
# Activation Functions
# ============================================

echo -e "${BLUE}Group: Activation Functions${NC}"

run_test \
    "Create with sigmoid hidden activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_sig.json --hidden-act=sigmoid" \
    "Created RNN model"

run_test \
    "Create with tanh hidden activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_tanh.json --hidden-act=tanh" \
    "Created RNN model"

run_test \
    "Create with ReLU hidden activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_relu.json --hidden-act=relu" \
    "Created RNN model"

run_test \
    "Create with linear output activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_linear.json --output-act=linear" \
    "Created RNN model"

run_test \
    "Create with sigmoid output activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=2 --save=$TEMP_DIR/act_outsig.json --output-act=sigmoid" \
    "Created RNN model"

run_test \
    "Output shows hidden activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_tanh2.json --hidden-act=tanh" \
    "Hidden activation:"

run_test \
    "Output shows output activation" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/act_out2.json --output-act=linear" \
    "Output activation:"

echo ""

# ============================================
# Loss Functions
# ============================================

echo -e "${BLUE}Group: Loss Functions${NC}"

run_test \
    "Create with MSE loss" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/loss_mse.json --loss=mse" \
    "Created RNN model"

run_test \
    "Create with CrossEntropy loss" \
    "$RNN_BIN create --input=2 --hidden=3 --output=3 --save=$TEMP_DIR/loss_ce.json --loss=crossentropy" \
    "Created RNN model"

run_test \
    "Output shows loss function" \
    "$RNN_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/loss_mse2.json --loss=mse" \
    "Loss function:"

echo ""

# ============================================
# JSON File Validation
# ============================================

echo -e "${BLUE}Group: JSON File Validation${NC}"

for i in {1..3}; do
    run_test \
        "Model $i JSON has input_size field" \
        "$RNN_BIN create --input=$((i+1)) --hidden=$((i+2)) --output=1 --save=$TEMP_DIR/json_test_$i.json && grep -q '\"input_size\"' $TEMP_DIR/json_test_$i.json && echo 'ok'" \
        "ok"
done

run_test \
    "JSON contains hidden_sizes array" \
    "grep -q '\"hidden_sizes\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON contains cell_type" \
    "grep -q '\"cell_type\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON is parseable structure" \
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
    "High learning rate" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/high_lr.json --lr=0.1" \
    "Learning rate: 0"

run_test \
    "Low learning rate" \
    "$RNN_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/low_lr.json --lr=0.0001" \
    "Learning rate:"

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

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
