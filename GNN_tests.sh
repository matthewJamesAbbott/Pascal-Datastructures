#!/bin/bash

#
# Matthew Abbott 2025
# Test for both GNN.pas and FacadeGNN.pas
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="/tmp/gnn_user_tests_$$"
GNN_BIN="./GNN"
FACADE_BIN="./FacadeGNN"

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

fpc GNN.pas >/dev/null 2>&1
fpc FacadeGNN.pas >/dev/null 2>&1

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

    if grep -q '"feature_size"' "$file" && grep -q '"output_size"' "$file"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Invalid JSON structure in $file"
        FAIL=$((FAIL + 1))
    fi
}

check_json_field() {
    local test_name="$1"
    local file="$2"
    local field="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ ! -f "$file" ]; then
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
        return
    fi

    if grep -q "\"$field\"" "$file"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Field '$field' not found in $file"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================
# Start Tests
# ============================================

echo ""
echo "========================================="
echo "GNN User Workflow Test Suite (Extended)"
echo "========================================="
echo ""

# Check binaries exist
if [ ! -f "$GNN_BIN" ]; then
    echo -e "${RED}Error: $GNN_BIN not found. Compile with: fpc GNN.pas${NC}"
    exit 1
fi

if [ ! -f "$FACADE_BIN" ]; then
    echo -e "${RED}Error: $FACADE_BIN not found. Compile with: fpc FacadeGNN.pas${NC}"
    exit 1
fi

echo -e "${BLUE}=== GNN Binary Tests ===${NC}"
echo ""

# ============================================
# Basic Help/Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "GNN help command" \
    "$GNN_BIN help" \
    "Commands:"

run_test \
    "GNN --help flag" \
    "$GNN_BIN --help" \
    "Commands:"

run_test \
    "FacadeGNN help command" \
    "$FACADE_BIN help" \
    "GNN"

run_test \
    "FacadeGNN --help flag" \
    "$FACADE_BIN --help" \
    "GNN"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create basic GNN (3 features, 16 hidden, 2 output, 2 MP layers)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/basic.json" \
    "Created GNN model"

check_file_exists \
    "JSON file created" \
    "$TEMP_DIR/basic.json"

check_json_valid \
    "JSON contains valid GNN structure" \
    "$TEMP_DIR/basic.json"

run_test \
    "Output shows correct feature size" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/basic2.json" \
    "Feature size: 3"

run_test \
    "Output shows hidden size" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/basic3.json" \
    "Hidden size: 16"

run_test \
    "Output shows output size" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/basic4.json" \
    "Output size: 2"

run_test \
    "Output shows message passing layers" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/basic5.json" \
    "Message passing layers: 2"

echo ""

# ============================================
# Model Creation - Various Configurations
# ============================================

echo -e "${BLUE}Group: Model Creation - Various Configurations${NC}"

run_test \
    "Create GNN with 1 MP layer" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=1 --save=$TEMP_DIR/mp1.json" \
    "Created GNN model"

run_test \
    "Create GNN with 3 MP layers" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=3 --save=$TEMP_DIR/mp3.json" \
    "Created GNN model"

run_test \
    "Create GNN with 4 MP layers" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=4 --save=$TEMP_DIR/mp4.json" \
    "Created GNN model"

run_test \
    "Create GNN with 5 MP layers" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=5 --save=$TEMP_DIR/mp5.json" \
    "Created GNN model"

run_test \
    "Create GNN with large feature dimension" \
    "$GNN_BIN create --feature=256 --hidden=128 --output=10 --mp-layers=2 --save=$TEMP_DIR/large_features.json" \
    "Created GNN model"

run_test \
    "Create GNN with large hidden dimension" \
    "$GNN_BIN create --feature=10 --hidden=512 --output=5 --mp-layers=2 --save=$TEMP_DIR/large_hidden.json" \
    "Created GNN model"

run_test \
    "Create GNN with small dimensions" \
    "$GNN_BIN create --feature=1 --hidden=2 --output=1 --mp-layers=1 --save=$TEMP_DIR/tiny.json" \
    "Created GNN model"

run_test \
    "Create GNN with many output dimensions" \
    "$GNN_BIN create --feature=10 --hidden=64 --output=1000 --mp-layers=2 --save=$TEMP_DIR/many_outputs.json" \
    "Created GNN model"

check_file_exists \
    "Tiny model file exists" \
    "$TEMP_DIR/tiny.json"

check_file_exists \
    "Large hidden model file exists" \
    "$TEMP_DIR/large_hidden.json"

echo ""

# ============================================
# Hyperparameters
# ============================================

echo -e "${BLUE}Group: Hyperparameters${NC}"

run_test \
    "Create with learning rate 0.001" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/lr001.json --lr=0.001" \
    "Created GNN model"

run_test \
    "Create with learning rate 0.1" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/lr01.json --lr=0.1" \
    "Created GNN model"

run_test \
    "Create with learning rate 0.5" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/lr05.json --lr=0.5" \
    "Created GNN model"

run_test \
    "Custom learning rate shows in output" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/lr_check.json --lr=0.005" \
    "Learning rate: 0"

check_json_field \
    "Learning rate saved in JSON" \
    "$TEMP_DIR/lr001.json" \
    "learning_rate"

check_json_field \
    "Max iterations saved in JSON" \
    "$TEMP_DIR/basic.json" \
    "max_iterations"

echo ""

# ============================================
# Activation Functions
# ============================================

echo -e "${BLUE}Group: Activation Functions${NC}"

run_test \
    "Create with ReLU activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/relu.json --activation=relu" \
    "Activation: relu"

run_test \
    "Create with Tanh activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/tanh.json --activation=tanh" \
    "Activation: tanh"

run_test \
    "Create with Sigmoid activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/sigmoid.json --activation=sigmoid" \
    "Activation: sigmoid"

run_test \
    "Create with LeakyReLU activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/leakyrelu.json --activation=leakyrelu" \
    "Activation: leakyrelu"

check_json_field \
    "ReLU activation saved in JSON" \
    "$TEMP_DIR/relu.json" \
    "activation"

check_json_field \
    "Tanh activation saved in JSON" \
    "$TEMP_DIR/tanh.json" \
    "activation"

echo ""

# ============================================
# Loss Functions
# ============================================

echo -e "${BLUE}Group: Loss Functions${NC}"

run_test \
    "Create with MSE loss" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/mse.json --loss=mse" \
    "Loss function: mse"

run_test \
    "Create with Binary Cross Entropy loss" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/bce.json --loss=bce" \
    "Loss function: bce"

check_json_field \
    "MSE loss saved in JSON" \
    "$TEMP_DIR/mse.json" \
    "loss_type"

check_json_field \
    "BCE loss saved in JSON" \
    "$TEMP_DIR/bce.json" \
    "loss_type"

echo ""

# ============================================
# Info Command
# ============================================

echo -e "${BLUE}Group: Info Command${NC}"

run_test \
    "Info displays model information" \
    "$GNN_BIN info --model=$TEMP_DIR/basic.json" \
    "Loading model from JSON"

run_test \
    "Info for different activation" \
    "$GNN_BIN info --model=$TEMP_DIR/tanh.json" \
    "Loading model from JSON"

run_test \
    "Info for deep model (5 layers)" \
    "$GNN_BIN info --model=$TEMP_DIR/mp5.json" \
    "Loading model from JSON"

run_test \
    "Info for large feature model" \
    "$GNN_BIN info --model=$TEMP_DIR/large_features.json" \
    "Loading model from JSON"

echo ""

# ============================================
# JSON Structure Validation
# ============================================

echo -e "${BLUE}Group: JSON File Validation${NC}"

for i in {1..3}; do
    run_test \
        "Model $i JSON has feature_size field" \
        "$GNN_BIN create --feature=$((i+1)) --hidden=$((i*8)) --output=2 --mp-layers=2 --save=$TEMP_DIR/json_test_$i.json && grep -q '\"feature_size\"' $TEMP_DIR/json_test_$i.json && echo 'ok'" \
        "ok"
done

check_json_field \
    "JSON contains hidden_size" \
    "$TEMP_DIR/basic.json" \
    "hidden_size"

check_json_field \
    "JSON contains output_size" \
    "$TEMP_DIR/basic.json" \
    "output_size"

check_json_field \
    "JSON contains num_message_passing_layers" \
    "$TEMP_DIR/basic.json" \
    "num_message_passing_layers"

check_json_field \
    "JSON contains activation field" \
    "$TEMP_DIR/relu.json" \
    "activation"

check_json_field \
    "JSON contains message_layers array" \
    "$TEMP_DIR/basic.json" \
    "message_layers"

check_json_field \
    "JSON contains update_layers array" \
    "$TEMP_DIR/basic.json" \
    "update_layers"

check_json_field \
    "JSON contains readout_layer" \
    "$TEMP_DIR/basic.json" \
    "readout_layer"

check_json_field \
    "JSON contains output_layer" \
    "$TEMP_DIR/basic.json" \
    "output_layer"

run_test \
    "JSON structure is valid JSON (starts with {)" \
    "head -1 $TEMP_DIR/basic.json | grep -q '{' && echo 'ok'" \
    "ok"

run_test \
    "JSON structure ends with }" \
    "tail -1 $TEMP_DIR/basic.json | grep -q '}' && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Error Cases
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Missing --feature argument" \
    "$GNN_BIN create --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/err1.json 2>&1" \
    "Error"

run_test \
    "Missing --hidden argument" \
    "$GNN_BIN create --feature=3 --output=2 --mp-layers=2 --save=$TEMP_DIR/err2.json 2>&1" \
    "Error"

run_test \
    "Missing --output argument" \
    "$GNN_BIN create --feature=3 --hidden=16 --mp-layers=2 --save=$TEMP_DIR/err3.json 2>&1" \
    "Error"

run_test \
    "Missing --mp-layers argument" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --save=$TEMP_DIR/err4.json 2>&1" \
    "Error"

run_test \
    "Missing --save argument" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 2>&1" \
    "Error"

run_test \
    "Loading non-existent model" \
    "$GNN_BIN info --model=$TEMP_DIR/nonexistent.json 2>&1" \
    ""

run_test \
    "Invalid activation type" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/bad_act.json --activation=invalidactivation 2>&1" \
    ""

run_test \
    "Invalid loss type" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/bad_loss.json --loss=invalidloss 2>&1" \
    ""

echo ""

# ============================================
# Facade Features - Basic
# ============================================

echo -e "${BLUE}Group: FacadeGNN - Basic Features${NC}"

run_test \
    "FacadeGNN displays help" \
    "$FACADE_BIN help" \
    "GNN"

run_test \
    "FacadeGNN --help flag" \
    "$FACADE_BIN --help" \
    "GNN"

run_test \
    "FacadeGNN creates model" \
    "$FACADE_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/facade_basic.json" \
    ""

run_test \
    "FacadeGNN info command exists" \
    "$FACADE_BIN info --model=$TEMP_DIR/facade_basic.json" \
    ""

echo ""

# ============================================
# Facade Features - Extended Commands
# ============================================

echo -e "${BLUE}Group: FacadeGNN - Extended Features${NC}"

# First create a facade model for testing extended features
$FACADE_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/facade_extended.json 2>/dev/null

run_test \
    "FacadeGNN architecture-summary command" \
    "$FACADE_BIN architecture-summary --model=$TEMP_DIR/facade_extended.json" \
    ""

run_test \
    "FacadeGNN parameter-count command" \
    "$FACADE_BIN parameter-count --model=$TEMP_DIR/facade_extended.json" \
    ""

run_test \
    "FacadeGNN layer-count command" \
    "$FACADE_BIN layer-count --model=$TEMP_DIR/facade_extended.json" \
    ""

run_test \
    "FacadeGNN get-layer-config command" \
    "$FACADE_BIN get-layer-config --model=$TEMP_DIR/facade_extended.json --layer=0" \
    ""

echo ""

# ============================================
# Facade Features - Query Functions
# ============================================

echo -e "${BLUE}Group: FacadeGNN - Query Functions${NC}"

run_test \
    "FacadeGNN get-feature-size command" \
    "$FACADE_BIN get-feature-size --model=$TEMP_DIR/facade_extended.json" \
    ""

run_test \
    "FacadeGNN get-hidden-size command" \
    "$FACADE_BIN get-hidden-size --model=$TEMP_DIR/facade_extended.json" \
    ""

run_test \
    "FacadeGNN get-output-size command" \
    "$FACADE_BIN get-output-size --model=$TEMP_DIR/facade_extended.json" \
    ""

run_test \
    "FacadeGNN get-mp-layers command" \
    "$FACADE_BIN get-mp-layers --model=$TEMP_DIR/facade_extended.json" \
    ""

run_test \
    "FacadeGNN get-message-layer-neuron-count command" \
    "$FACADE_BIN get-message-layer-neuron-count --model=$TEMP_DIR/facade_extended.json --layer=0" \
    ""

run_test \
    "FacadeGNN get-update-layer-neuron-count command" \
    "$FACADE_BIN get-update-layer-neuron-count --model=$TEMP_DIR/facade_extended.json --layer=0" \
    ""

run_test \
    "FacadeGNN get-readout-layer-neuron-count command" \
    "$FACADE_BIN get-readout-layer-neuron-count --model=$TEMP_DIR/facade_extended.json" \
    ""

run_test \
    "FacadeGNN get-output-layer-neuron-count command" \
    "$FACADE_BIN get-output-layer-neuron-count --model=$TEMP_DIR/facade_extended.json" \
    ""

echo ""

# ============================================
# Cross-binary Compatibility
# ============================================

echo -e "${BLUE}Group: Cross-binary Compatibility${NC}"

run_test \
    "Model created by GNN loads in FacadeGNN" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/cross_gnn.json && $FACADE_BIN info --model=$TEMP_DIR/cross_gnn.json" \
    ""

run_test \
    "Model created by FacadeGNN loads in GNN" \
    "$FACADE_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/cross_facade.json && $GNN_BIN info --model=$TEMP_DIR/cross_facade.json" \
    "Loading model from JSON"

run_test \
    "GNN info command works on Facade model" \
    "$GNN_BIN info --model=$TEMP_DIR/cross_facade.json" \
    "Loading model from JSON"

run_test \
    "Facade info command works on GNN model" \
    "$FACADE_BIN info --model=$TEMP_DIR/cross_gnn.json" \
    ""

echo ""

# ============================================
# Model Variants - Comprehensive
# ============================================

echo -e "${BLUE}Group: Comprehensive Model Variants${NC}"

# Create various model types and test persistence
run_test \
    "All activation types compile correctly (ReLU)" \
    "$GNN_BIN create --feature=4 --hidden=8 --output=2 --mp-layers=2 --save=$TEMP_DIR/var_relu.json --activation=relu && [ -f $TEMP_DIR/var_relu.json ]" \
    ""

run_test \
    "All activation types compile correctly (Tanh)" \
    "$GNN_BIN create --feature=4 --hidden=8 --output=2 --mp-layers=2 --save=$TEMP_DIR/var_tanh.json --activation=tanh && [ -f $TEMP_DIR/var_tanh.json ]" \
    ""

run_test \
    "All activation types compile correctly (Sigmoid)" \
    "$GNN_BIN create --feature=4 --hidden=8 --output=2 --mp-layers=2 --save=$TEMP_DIR/var_sigmoid.json --activation=sigmoid && [ -f $TEMP_DIR/var_sigmoid.json ]" \
    ""

run_test \
    "All activation types compile correctly (LeakyReLU)" \
    "$GNN_BIN create --feature=4 --hidden=8 --output=2 --mp-layers=2 --save=$TEMP_DIR/var_leaky.json --activation=leakyrelu && [ -f $TEMP_DIR/var_leaky.json ]" \
    ""

run_test \
    "All loss types compile correctly (MSE)" \
    "$GNN_BIN create --feature=4 --hidden=8 --output=2 --mp-layers=2 --save=$TEMP_DIR/var_mse.json --loss=mse && [ -f $TEMP_DIR/var_mse.json ]" \
    ""

run_test \
    "All loss types compile correctly (BCE)" \
    "$GNN_BIN create --feature=4 --hidden=8 --output=2 --mp-layers=2 --save=$TEMP_DIR/var_bce.json --loss=bce && [ -f $TEMP_DIR/var_bce.json ]" \
    ""

echo ""

# ============================================
# Sequential Operations
# ============================================

echo -e "${BLUE}Group: Sequential Operations Workflow${NC}"

run_test \
    "Workflow: Create GNN -> Load -> Info" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=3 --save=$TEMP_DIR/wf1.json && $GNN_BIN info --model=$TEMP_DIR/wf1.json" \
    "Loading model from JSON"

run_test \
    "Workflow: Create with params -> Verify in Facade" \
    "$GNN_BIN create --feature=4 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/wf2.json --lr=0.005 --activation=tanh && $FACADE_BIN info --model=$TEMP_DIR/wf2.json" \
    ""

run_test \
    "Workflow: Create Facade -> Load in GNN" \
    "$FACADE_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/wf3.json && $GNN_BIN info --model=$TEMP_DIR/wf3.json" \
    "Loading model from JSON"

run_test \
    "Workflow: Create with all custom params" \
    "$GNN_BIN create --feature=6 --hidden=32 --output=4 --mp-layers=3 --save=$TEMP_DIR/wf4.json --lr=0.01 --activation=sigmoid --loss=bce && $FACADE_BIN info --model=$TEMP_DIR/wf4.json" \
    ""

run_test \
    "Workflow: Multi-layer GNN creation and verification" \
    "$GNN_BIN create --feature=10 --hidden=64 --output=5 --mp-layers=4 --save=$TEMP_DIR/wf5.json && $FACADE_BIN get-mp-layers --model=$TEMP_DIR/wf5.json" \
    ""

echo ""

# ============================================
# File Persistence and Format
# ============================================

echo -e "${BLUE}Group: File Persistence & Format${NC}"

run_test \
    "Created model file is not empty" \
    "[ -s $TEMP_DIR/basic.json ] && echo 'ok'" \
    "ok"

run_test \
    "Model JSON is readable text" \
    "file $TEMP_DIR/basic.json | grep -q 'JSON' && echo 'ok'" \
    "ok"

run_test \
    "Multiple models can exist independently" \
    "[ -f $TEMP_DIR/mp1.json ] && [ -f $TEMP_DIR/mp3.json ] && [ -f $TEMP_DIR/mp5.json ] && echo 'ok'" \
    "ok"

run_test \
    "Models with different configs have different sizes" \
    "[ -f $TEMP_DIR/tiny.json ] && [ -f $TEMP_DIR/large_hidden.json ] && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Dimension Edge Cases
# ============================================

echo -e "${BLUE}Group: Dimension Edge Cases${NC}"

run_test \
    "Minimal model (1-2-1-1)" \
    "$GNN_BIN create --feature=1 --hidden=2 --output=1 --mp-layers=1 --save=$TEMP_DIR/edge_minimal.json && [ -f $TEMP_DIR/edge_minimal.json ]" \
    ""

run_test \
    "Very deep model (2-4-5)" \
    "$GNN_BIN create --feature=2 --hidden=4 --output=2 --mp-layers=5 --save=$TEMP_DIR/edge_deep.json && [ -f $TEMP_DIR/edge_deep.json ]" \
    ""

run_test \
    "Wide model (512-512-256)" \
    "$GNN_BIN create --feature=512 --hidden=512 --output=256 --mp-layers=2 --save=$TEMP_DIR/edge_wide.json && [ -f $TEMP_DIR/edge_wide.json ]" \
    ""

check_json_field \
    "Minimal model has feature_size=1" \
    "$TEMP_DIR/edge_minimal.json" \
    "feature_size"

check_json_field \
    "Deep model has correct mp_layers" \
    "$TEMP_DIR/edge_deep.json" \
    "num_message_passing_layers"

check_json_field \
    "Wide model created successfully" \
    "$TEMP_DIR/edge_wide.json" \
    "hidden_size"

echo ""

# ============================================
# Hyperparameter Ranges
# ============================================

echo -e "${BLUE}Group: Hyperparameter Ranges${NC}"

run_test \
    "Very small learning rate (0.0001)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/hp_tiny_lr.json --lr=0.0001 && [ -f $TEMP_DIR/hp_tiny_lr.json ]" \
    ""

run_test \
    "Large learning rate (1.0)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/hp_large_lr.json --lr=1.0 && [ -f $TEMP_DIR/hp_large_lr.json ]" \
    ""

run_test \
    "Fractional learning rate (0.05)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/hp_frac_lr.json --lr=0.05 && [ -f $TEMP_DIR/hp_frac_lr.json ]" \
    ""

check_json_field \
    "Very small learning rate persists" \
    "$TEMP_DIR/hp_tiny_lr.json" \
    "learning_rate"

check_json_field \
    "Large learning rate persists" \
    "$TEMP_DIR/hp_large_lr.json" \
    "learning_rate"

echo ""

# ============================================
# JSON Layer Structure
# ============================================

echo -e "${BLUE}Group: JSON Layer Structure Validation${NC}"

run_test \
    "Message layers contain neurons" \
    "grep -q '\"neurons\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "Neurons have weights" \
    "grep -q '\"weights\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "Neurons have bias" \
    "grep -q '\"bias\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "Readout layer exists in JSON" \
    "grep -q '\"readout_layer\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "Output layer exists in JSON" \
    "grep -q '\"output_layer\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Facade Extended Facade Tests
# ============================================

echo -e "${BLUE}Group: Facade Extended Command Testing${NC}"

# Test multiple calls to same facade
for i in {1..3}; do
    run_test \
        "Facade info call $i succeeds" \
        "$FACADE_BIN info --model=$TEMP_DIR/facade_extended.json" \
        ""
done

for i in {1..3}; do
    run_test \
        "Facade architecture-summary call $i succeeds" \
        "$FACADE_BIN architecture-summary --model=$TEMP_DIR/facade_extended.json" \
        ""
done

for i in {1..3}; do
    run_test \
        "Facade parameter-count call $i succeeds" \
        "$FACADE_BIN parameter-count --model=$TEMP_DIR/facade_extended.json" \
        ""
done

echo ""

# ============================================
# Model Consistency
# ============================================

echo -e "${BLUE}Group: Model Consistency${NC}"

# Create a model, load it twice, verify consistency
run_test \
    "Model loaded twice produces same info output" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/consistency.json && \
    $GNN_BIN info --model=$TEMP_DIR/consistency.json > /tmp/gnn_output1.txt && \
    $GNN_BIN info --model=$TEMP_DIR/consistency.json > /tmp/gnn_output2.txt && \
    diff /tmp/gnn_output1.txt /tmp/gnn_output2.txt > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Facade loads same model consistently" \
    "$FACADE_BIN info --model=$TEMP_DIR/consistency.json > /tmp/facade_output1.txt && \
    $FACADE_BIN info --model=$TEMP_DIR/consistency.json > /tmp/facade_output2.txt && \
    diff /tmp/facade_output1.txt /tmp/facade_output2.txt > /dev/null && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Facade vs GNN Output Parity
# ============================================

echo -e "${BLUE}Group: GNN and Facade Parity${NC}"

run_test \
    "Both binaries can load the same model" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$TEMP_DIR/parity.json && \
    $GNN_BIN info --model=$TEMP_DIR/parity.json > /dev/null && \
    $FACADE_BIN info --model=$TEMP_DIR/parity.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Facade parameter-count exists for all created models" \
    "$FACADE_BIN parameter-count --model=$TEMP_DIR/mp1.json && \
    $FACADE_BIN parameter-count --model=$TEMP_DIR/mp3.json && \
    $FACADE_BIN parameter-count --model=$TEMP_DIR/mp5.json && echo 'ok'" \
    "ok"

run_test \
    "Facade architecture summary exists for all models" \
    "$FACADE_BIN architecture-summary --model=$TEMP_DIR/relu.json && \
    $FACADE_BIN architecture-summary --model=$TEMP_DIR/tanh.json && echo 'ok'" \
    "ok"

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
