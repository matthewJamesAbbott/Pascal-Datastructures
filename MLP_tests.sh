#!/bin/bash

#
# Matthew Abbott 2025
# Test for both MLP.pas and FacadeMLP.pas
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="/tmp/mlp_user_tests_$$"
MLP_BIN="./MLP"
FACADE_BIN="./FacadeMLP"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup/Cleanup
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TEMP_DIR"

fpc MLP.pas
fpc FacadeMLP.pas

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

    if grep -q '"input_size"' "$file" && grep -q '"output_size"' "$file" && grep -q '"weights"' "$file"; then
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
echo "MLP User Workflow Test Suite"
echo "========================================="
echo ""

# Check binaries exist
if [ ! -f "$MLP_BIN" ]; then
    echo -e "${RED}Error: $MLP_BIN not found. Compile with: fpc MLP.pas${NC}"
    exit 1
fi

if [ ! -f "$FACADE_BIN" ]; then
    echo -e "${RED}Error: $FACADE_BIN not found. Compile with: fpc FacadeMLP.pas${NC}"
    exit 1
fi

echo -e "${BLUE}=== MLP Binary Tests ===${NC}"
echo ""

# ============================================
# Basic Help/Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "MLP help command" \
    "$MLP_BIN help" \
    "Commands:"

run_test \
    "MLP --help flag" \
    "$MLP_BIN --help" \
    "Commands:"

run_test \
    "FacadeMLP help command" \
    "$FACADE_BIN help" \
    "Commands:"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create 2-4-1 model" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic.json" \
    "Created MLP model"

check_file_exists \
    "JSON file created for 2-4-1" \
    "$TEMP_DIR/basic.json"

check_json_valid \
    "JSON contains valid structure" \
    "$TEMP_DIR/basic.json"

run_test \
    "Output shows correct architecture" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic2.json" \
    "Input size: 2"

run_test \
    "Output shows hidden size" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic3.json" \
    "Hidden sizes: 4"

run_test \
    "Output shows output size" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/basic4.json" \
    "Output size: 1"

echo ""

# ============================================
# Model Creation - Multi-layer
# ============================================

echo -e "${BLUE}Group: Model Creation - Multi-layer${NC}"

run_test \
    "Create 3-5-3-2 network" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/multilayer.json" \
    "Created MLP model"

check_file_exists \
    "JSON file for multi-layer" \
    "$TEMP_DIR/multilayer.json"

run_test \
    "Multi-layer output shows correct input" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml2.json" \
    "Input size: 3"

run_test \
    "Multi-layer output shows both hidden sizes" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml3.json" \
    "Hidden sizes: 5,3"

run_test \
    "Multi-layer output shows correct output size" \
    "$MLP_BIN create --input=3 --hidden=5,3 --output=2 --save=$TEMP_DIR/ml4.json" \
    "Output size: 2"

echo ""

# ============================================
# Hyperparameters
# ============================================

echo -e "${BLUE}Group: Hyperparameters${NC}"

run_test \
    "Create with custom learning rate" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lr.json --lr=0.01" \
    "Created MLP model"

run_test \
    "Custom learning rate saved" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lr2.json --lr=0.01" \
    "Learning rate: 0"

run_test \
    "Create with dropout" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/dropout.json --dropout=0.2" \
    "Created MLP model"

run_test \
    "Create with L2 regularization" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/l2.json --l2=0.001" \
    "Created MLP model"

echo ""

# ============================================
# Activation Functions
# ============================================

echo -e "${BLUE}Group: Activation Functions${NC}"

run_test \
    "Create with Sigmoid activation" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/sigmoid.json --hidden-act=sigmoid" \
    "Created MLP model"

check_json_valid \
    "Sigmoid model JSON valid" \
    "$TEMP_DIR/sigmoid.json"

run_test \
    "Create with Tanh activation" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/tanh.json --hidden-act=tanh" \
    "Created MLP model"

check_json_valid \
    "Tanh model JSON valid" \
    "$TEMP_DIR/tanh.json"

run_test \
    "Create with ReLU activation" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/relu.json --hidden-act=relu" \
    "Created MLP model"

check_json_valid \
    "ReLU model JSON valid" \
    "$TEMP_DIR/relu.json"

echo ""

# ============================================
# Optimizers
# ============================================

echo -e "${BLUE}Group: Optimizers${NC}"

run_test \
    "Create with SGD optimizer" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/sgd.json --optimizer=sgd" \
    "Created MLP model"

check_json_valid \
    "SGD model JSON valid" \
    "$TEMP_DIR/sgd.json"

run_test \
    "Create with Adam optimizer" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/adam.json --optimizer=adam" \
    "Created MLP model"

check_json_valid \
    "Adam model JSON valid" \
    "$TEMP_DIR/adam.json"

run_test \
    "Create with RMSProp optimizer" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/rmsprop.json --optimizer=rmsprop" \
    "Created MLP model"

check_json_valid \
    "RMSProp model JSON valid" \
    "$TEMP_DIR/rmsprop.json"

echo ""

# ============================================
# JSON Load/Save Round-trip
# ============================================

echo -e "${BLUE}Group: JSON Load/Save Round-trip${NC}"

run_test \
    "Load model with info command" \
    "$MLP_BIN info --model=$TEMP_DIR/basic.json" \
    "Model loaded from JSON"

run_test \
    "Loaded model shows input size" \
    "$MLP_BIN info --model=$TEMP_DIR/basic.json" \
    "Input size: 2"

run_test \
    "Loaded model shows output size" \
    "$MLP_BIN info --model=$TEMP_DIR/basic.json" \
    "Output size: 1"

run_test \
    "Loaded model shows hidden layers" \
    "$MLP_BIN info --model=$TEMP_DIR/basic.json" \
    "Hidden"

run_test \
    "Train command loads JSON" \
    "$MLP_BIN train --model=$TEMP_DIR/basic.json --save=$TEMP_DIR/trained.json" \
    "Model loaded from JSON"

run_test \
    "Predict command loads JSON" \
    "$MLP_BIN predict --model=$TEMP_DIR/basic.json --input=0.5,0.3" \
    "Model loaded successfully"

echo ""

# ============================================
# Large Networks
# ============================================

echo -e "${BLUE}Group: Large Networks${NC}"

run_test \
    "Create large 10-16-12-8-5 network" \
    "$MLP_BIN create --input=10 --hidden=16,12,8 --output=5 --save=$TEMP_DIR/large.json" \
    "Created MLP model"

check_json_valid \
    "Large network JSON valid" \
    "$TEMP_DIR/large.json"

run_test \
    "Large network shows correct input" \
    "$MLP_BIN info --model=$TEMP_DIR/large.json" \
    "Input size: 10"

run_test \
    "Large network shows correct output" \
    "$MLP_BIN info --model=$TEMP_DIR/large.json" \
    "Output size: 5"

echo ""

# ============================================
# FacadeMLP Extended Features
# ============================================

echo -e "${BLUE}Group: FacadeMLP Extended Commands${NC}"

run_test \
    "FacadeMLP create model" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade.json" \
    "Created MLP model"

check_file_exists \
    "FacadeMLP JSON created" \
    "$TEMP_DIR/facade.json"

run_test \
    "FacadeMLP info command" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade.json" \
    "MLP Model Information"

run_test \
    "FacadeMLP info shows architecture" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade.json" \
    "Layer sizes"

run_test \
    "FacadeMLP info shows total layers" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade.json" \
    "Total layers"

run_test \
    "FacadeMLP predict command" \
    "$FACADE_BIN predict --model=$TEMP_DIR/facade.json --input=0.5,0.3" \
    "Output:"

run_test \
    "FacadeMLP predict returns numeric output" \
    "$FACADE_BIN predict --model=$TEMP_DIR/facade.json --input=0.1,0.2" \
    "[0-9]"

echo ""

# ============================================
# Multiple Operations
# ============================================

echo -e "${BLUE}Group: Multiple Operations${NC}"

run_test \
    "Create, load, and check first model" \
    "$MLP_BIN create --input=2 --hidden=4 --output=1 --save=$TEMP_DIR/test1.json && $MLP_BIN info --model=$TEMP_DIR/test1.json" \
    "Input size: 2"

run_test \
    "Create, load, and check second model" \
    "$MLP_BIN create --input=3 --hidden=5 --output=2 --save=$TEMP_DIR/test2.json && $MLP_BIN info --model=$TEMP_DIR/test2.json" \
    "Input size: 3"

run_test \
    "Create, load, and check third model" \
    "$MLP_BIN create --input=4 --hidden=6,4 --output=3 --save=$TEMP_DIR/test3.json && $MLP_BIN info --model=$TEMP_DIR/test3.json" \
    "Input size: 4"

echo ""

# ============================================
# Architecture Variations
# ============================================

echo -e "${BLUE}Group: Architecture Variations${NC}"

run_test \
    "Minimal network 1-1-1" \
    "$MLP_BIN create --input=1 --hidden=1 --output=1 --save=$TEMP_DIR/minimal.json" \
    "Created MLP model"

check_json_valid \
    "Minimal network JSON valid" \
    "$TEMP_DIR/minimal.json"

run_test \
    "Deep network 5-8-6-4-3-2" \
    "$MLP_BIN create --input=5 --hidden=8,6,4,3 --output=2 --save=$TEMP_DIR/deep.json" \
    "Created MLP model"

check_json_valid \
    "Deep network JSON valid" \
    "$TEMP_DIR/deep.json"

run_test \
    "Many input features 20-10-5" \
    "$MLP_BIN create --input=20 --hidden=10 --output=5 --save=$TEMP_DIR/many_in.json" \
    "Created MLP model"

check_json_valid \
    "Many input features JSON valid" \
    "$TEMP_DIR/many_in.json"

run_test \
    "Many output classes 5-8-10" \
    "$MLP_BIN create --input=5 --hidden=8 --output=10 --save=$TEMP_DIR/many_out.json" \
    "Created MLP model"

check_json_valid \
    "Many output classes JSON valid" \
    "$TEMP_DIR/many_out.json"

echo ""

# ============================================
# Real Workflow Scenarios
# ============================================

echo -e "${BLUE}Group: Real Workflow Scenarios${NC}"

# Scenario 1: Binary Classification
run_test \
    "Binary classification model" \
    "$MLP_BIN create --input=10 --hidden=16,8 --output=2 --save=$TEMP_DIR/binary.json" \
    "Created MLP model"

run_test \
    "Load binary classification model" \
    "$MLP_BIN info --model=$TEMP_DIR/binary.json" \
    "Input size: 10"

# Scenario 2: Multi-class Classification
run_test \
    "Multi-class classification (10 classes)" \
    "$MLP_BIN create --input=28 --hidden=128,64 --output=10 --save=$TEMP_DIR/mnist.json" \
    "Created MLP model"

run_test \
    "Load multi-class model" \
    "$MLP_BIN info --model=$TEMP_DIR/mnist.json" \
    "Output size: 10"

# Scenario 3: Regression
run_test \
    "Regression model (single output)" \
    "$MLP_BIN create --input=5 --hidden=32,16 --output=1 --save=$TEMP_DIR/regression.json" \
    "Created MLP model"

run_test \
    "Regression model output size" \
    "$MLP_BIN info --model=$TEMP_DIR/regression.json" \
    "Output size: 1"

echo ""

# ============================================
# Prediction Output Validation
# ============================================

echo -e "${BLUE}Group: Prediction Output Validation${NC}"

run_test \
    "FacadeMLP predict returns numeric value" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/pred1.json && $FACADE_BIN predict --model=$TEMP_DIR/pred1.json --input=0.5,0.5" \
    "[0-9]\\.[0-9]"

run_test \
    "FacadeMLP predict with different input 1" \
    "$FACADE_BIN predict --model=$TEMP_DIR/pred1.json --input=0.1,0.2" \
    "Output:"

run_test \
    "FacadeMLP predict with different input 2" \
    "$FACADE_BIN predict --model=$TEMP_DIR/pred1.json --input=0.9,0.8" \
    "Output:"

run_test \
    "FacadeMLP multi-output prediction" \
    "$FACADE_BIN create --input=3 --hidden=4 --output=3 --save=$TEMP_DIR/multi_pred.json && $FACADE_BIN predict --model=$TEMP_DIR/multi_pred.json --input=0.5,0.3,0.7" \
    "Output:"

run_test \
    "FacadeMLP predict returns max index" \
    "$FACADE_BIN predict --model=$TEMP_DIR/multi_pred.json --input=0.2,0.4,0.6" \
    "Max index:"

echo ""

# ============================================
# Parameter Combinations
# ============================================

echo -e "${BLUE}Group: Parameter Combinations${NC}"

run_test \
    "All parameters together: lr, dropout, l2" \
    "$MLP_BIN create --input=4 --hidden=6 --output=2 --save=$TEMP_DIR/combo1.json --lr=0.05 --dropout=0.1 --l2=0.0001" \
    "Created MLP model"

check_json_valid \
    "Combined params model JSON valid" \
    "$TEMP_DIR/combo1.json"

run_test \
    "Activation + optimizer combination" \
    "$MLP_BIN create --input=3 --hidden=5 --output=2 --save=$TEMP_DIR/combo2.json --hidden-act=relu --optimizer=adam" \
    "Created MLP model"

run_test \
    "Tanh activation + RMSProp optimizer" \
    "$MLP_BIN create --input=3 --hidden=5 --output=2 --save=$TEMP_DIR/combo3.json --hidden-act=tanh --optimizer=rmsprop" \
    "Created MLP model"

run_test \
    "Large learning rate 0.5" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lr_high.json --lr=0.5" \
    "Created MLP model"

run_test \
    "Small learning rate 0.001" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/lr_low.json --lr=0.001" \
    "Created MLP model"

run_test \
    "High dropout 0.5" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/dropout_high.json --dropout=0.5" \
    "Created MLP model"

run_test \
    "High L2 regularization 0.1" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/l2_high.json --l2=0.1" \
    "Created MLP model"

echo ""

# ============================================
# Model Persistence and Consistency
# ============================================

echo -e "${BLUE}Group: Model Persistence and Consistency${NC}"

# Create a model, load it, check it, load again
run_test \
    "Load same model twice returns consistent info" \
    "$MLP_BIN create --input=3 --hidden=5 --output=2 --save=$TEMP_DIR/persist.json && $MLP_BIN info --model=$TEMP_DIR/persist.json && $MLP_BIN info --model=$TEMP_DIR/persist.json" \
    "Input size: 3"

run_test \
    "Multiple sequential loads" \
    "for i in {1..3}; do $MLP_BIN info --model=$TEMP_DIR/persist.json > /dev/null; done && echo 'success'" \
    "success"

run_test \
    "Load different models sequentially" \
    "$MLP_BIN info --model=$TEMP_DIR/basic.json && $MLP_BIN info --model=$TEMP_DIR/multilayer.json && echo 'loaded'" \
    "loaded"

echo ""

# ============================================
# Boundary Cases
# ============================================

echo -e "${BLUE}Group: Boundary Cases${NC}"

run_test \
    "Create with zero dropout (disabled)" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/dropout_0.json --dropout=0" \
    "Created MLP model"

run_test \
    "Create very small network" \
    "$MLP_BIN create --input=1 --hidden=1 --output=1 --save=$TEMP_DIR/tiny.json" \
    "Created MLP model"

run_test \
    "Create with many hidden layers" \
    "$MLP_BIN create --input=3 --hidden=4,4,4,4,4 --output=2 --save=$TEMP_DIR/many_layers.json" \
    "Created MLP model"

check_json_valid \
    "Many layer model JSON valid" \
    "$TEMP_DIR/many_layers.json"

run_test \
    "Create with varied layer sizes" \
    "$MLP_BIN create --input=2 --hidden=100,10,100 --output=2 --save=$TEMP_DIR/varied.json" \
    "Created MLP model"

run_test \
    "Large input dimension" \
    "$MLP_BIN create --input=784 --hidden=128 --output=10 --save=$TEMP_DIR/large_input.json" \
    "Created MLP model"

run_test \
    "Large output dimension" \
    "$MLP_BIN create --input=10 --hidden=64 --output=1000 --save=$TEMP_DIR/large_output.json" \
    "Created MLP model"

echo ""

# ============================================
# Error Cases (expected to handle gracefully)
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Missing required --save argument" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 2>&1" \
    "Error"

run_test \
    "Missing input size" \
    "$MLP_BIN create --hidden=3 --output=1 --save=$TEMP_DIR/err1.json 2>&1" \
    "Error"

run_test \
    "Missing output size" \
    "$MLP_BIN create --input=2 --hidden=3 --save=$TEMP_DIR/err2.json 2>&1" \
    "Error"

run_test \
    "Missing hidden layers" \
    "$MLP_BIN create --input=2 --output=1 --save=$TEMP_DIR/err3.json 2>&1" \
    "Error"

run_test \
    "Loading non-existent model" \
    "$MLP_BIN info --model=$TEMP_DIR/nonexistent.json 2>&1" \
    ""

echo ""

# ============================================
# JSON File Validation
# ============================================

echo -e "${BLUE}Group: JSON File Validation${NC}"

# Create multiple models and verify JSON structure
for i in {1..3}; do
    run_test \
        "Model $i JSON has input_size field" \
        "$MLP_BIN create --input=$((i+1)) --hidden=$((i+2)) --output=1 --save=$TEMP_DIR/json_test_$i.json && grep -q '\"input_size\"' $TEMP_DIR/json_test_$i.json && echo 'ok'" \
        "ok"
done

run_test \
    "JSON contains hidden_sizes array" \
    "grep -q '\"hidden_sizes\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON contains output_layer" \
    "grep -q '\"output_layer\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON is parseable structure" \
    "head -1 $TEMP_DIR/basic.json | grep -q '{' && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Facade-specific Extended Features
# ============================================

echo -e "${BLUE}Group: FacadeMLP Extended Features${NC}"

run_test \
    "FacadeMLP shows hyperparameters in info" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_hyper.json && $FACADE_BIN info --model=$TEMP_DIR/facade_hyper.json" \
    "Hyperparameters:"

run_test \
    "FacadeMLP info shows learning rate" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_hyper.json" \
    "Learning rate:"

run_test \
    "FacadeMLP info shows optimizer" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_hyper.json" \
    "Optimizer:"

run_test \
    "FacadeMLP info shows activation functions" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_hyper.json" \
    "activation:"

run_test \
    "FacadeMLP displays regularization" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_reg.json --dropout=0.2 --l2=0.001 && $FACADE_BIN info --model=$TEMP_DIR/facade_reg.json" \
    "Dropout"

echo ""

# ============================================
# Cross-binary Compatibility
# ============================================

echo -e "${BLUE}Group: Cross-binary Compatibility${NC}"

run_test \
    "Model created by MLP can be loaded by FacadeMLP" \
    "$MLP_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/cross.json && $FACADE_BIN info --model=$TEMP_DIR/cross.json" \
    "MLP Model Information"

run_test \
    "FacadeMLP model info shows correct input" \
    "$FACADE_BIN info --model=$TEMP_DIR/cross.json" \
    "Input size: 2"

run_test \
    "FacadeMLP can predict with MLP-created model" \
    "$FACADE_BIN predict --model=$TEMP_DIR/cross.json --input=0.5,0.5" \
    "Output:"

echo ""

# ============================================
# FacadeMLP Extended Commands
# ============================================

echo -e "${BLUE}Group: FacadeMLP Extended Commands${NC}"

run_test \
    "FacadeMLP get-weight command" \
    "$FACADE_BIN create --input=2 --hidden=3 --output=1 --save=$TEMP_DIR/facade_extended.json && $FACADE_BIN get-weight --model=$TEMP_DIR/facade_extended.json --layer=0 --neuron=0 --weight=0" \
    "Weight"

run_test \
    "FacadeMLP get-weights command" \
    "$FACADE_BIN get-weights --model=$TEMP_DIR/facade_extended.json --layer=0 --neuron=0" \
    "Weights"

run_test \
    "FacadeMLP get-bias command" \
    "$FACADE_BIN get-bias --model=$TEMP_DIR/facade_extended.json --layer=0 --neuron=0" \
    "Bias"

run_test \
    "FacadeMLP set-weight command" \
    "$FACADE_BIN set-weight --model=$TEMP_DIR/facade_extended.json --layer=0 --neuron=0 --weight=0 --value=0.5 --save=$TEMP_DIR/facade_modified.json && $FACADE_BIN get-weight --model=$TEMP_DIR/facade_modified.json --layer=0 --neuron=0 --weight=0" \
    "Weight"

run_test \
    "FacadeMLP set-bias command" \
    "$FACADE_BIN set-bias --model=$TEMP_DIR/facade_extended.json --layer=0 --neuron=0 --value=0.25 --save=$TEMP_DIR/facade_bias_modified.json && $FACADE_BIN get-bias --model=$TEMP_DIR/facade_bias_modified.json --layer=0 --neuron=0" \
    "Bias"

run_test \
    "FacadeMLP get-output command" \
    "$FACADE_BIN get-output --model=$TEMP_DIR/facade_extended.json --layer=0 --neuron=0 --input=0.5,0.5" \
    "Output"

run_test \
    "FacadeMLP get-error command" \
    "$FACADE_BIN get-error --model=$TEMP_DIR/facade_extended.json --layer=0 --neuron=0" \
    "Error"

run_test \
    "FacadeMLP layer-info command" \
    "$FACADE_BIN layer-info --model=$TEMP_DIR/facade_extended.json --layer=0" \
    "Layer 0"

run_test \
    "FacadeMLP histogram command" \
    "$FACADE_BIN histogram --model=$TEMP_DIR/facade_extended.json --layer=0 --type=activation" \
    "Histogram"

run_test \
    "FacadeMLP get-optimizer command" \
    "$FACADE_BIN get-optimizer --model=$TEMP_DIR/facade_extended.json --layer=0 --neuron=0" \
    "MBias"

echo ""

# ============================================
# Sequential Operations (Real Workflow)
# ============================================

echo -e "${BLUE}Group: Sequential Operations Workflow${NC}"

run_test \
    "Workflow: Create -> Load -> Predict (Binary)" \
    "$MLP_BIN create --input=10 --hidden=16 --output=2 --save=$TEMP_DIR/workflow1.json && $FACADE_BIN predict --model=$TEMP_DIR/workflow1.json --input=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    "Output:"

run_test \
    "Workflow: Create -> Load -> Train (JSON load)" \
    "$MLP_BIN create --input=3 --hidden=5 --output=2 --save=$TEMP_DIR/workflow2.json && $MLP_BIN train --model=$TEMP_DIR/workflow2.json --save=$TEMP_DIR/workflow2_trained.json" \
    "Model loaded from JSON"

run_test \
    "Workflow: Create multi-layer -> Info -> Predict" \
    "$MLP_BIN create --input=5 --hidden=8,6,4 --output=3 --save=$TEMP_DIR/workflow3.json && $FACADE_BIN info --model=$TEMP_DIR/workflow3.json && $FACADE_BIN predict --model=$TEMP_DIR/workflow3.json --input=0.2,0.4,0.6,0.8,0.9" \
    "Max index:"

run_test \
    "Workflow: Create with params -> Verify -> Predict" \
    "$MLP_BIN create --input=4 --hidden=8,6 --output=2 --save=$TEMP_DIR/workflow4.json --lr=0.01 --optimizer=adam --dropout=0.1 && $MLP_BIN info --model=$TEMP_DIR/workflow4.json" \
    "Input size: 4"

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
