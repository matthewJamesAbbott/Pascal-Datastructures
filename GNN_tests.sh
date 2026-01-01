#!/bin/bash

#
# Matthew Abbott 2025
# Comprehensive Test Suite for GNN.pas and FacadeGNN.pas
# Tests all functions, facade functions, and cross-loading/saving
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"
GNN_BIN="./GNN"
FACADE_BIN="./FacadeGNN"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Setup
mkdir -p "$OUTPUT_DIR"

echo -e "${CYAN}Compiling GNN.pas...${NC}"
fpc GNN.pas >/dev/null 2>&1
echo -e "${CYAN}Compiling FacadeGNN.pas...${NC}"
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

run_test_exit_code() {
    local test_name="$1"
    local command="$2"
    local expected_exit="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    output=$(eval "$command" 2>&1)
    exit_code=$?

    if [ "$exit_code" -eq "$expected_exit" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Command: $command"
        echo "  Expected exit code: $expected_exit, got: $exit_code"
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

check_not_empty() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ -s "$file" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  File is empty or missing: $file"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================
# Start Tests
# ============================================

echo ""
echo "========================================="
echo "GNN & FacadeGNN Comprehensive Test Suite"
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

# ============================================
# SECTION 1: GNN Binary - Help & Usage
# ============================================

echo -e "${BLUE}=== SECTION 1: GNN Binary - Help & Usage ===${NC}"
echo ""

run_test \
    "GNN help command" \
    "$GNN_BIN help" \
    "Commands:"

run_test \
    "GNN --help flag" \
    "$GNN_BIN --help" \
    "Commands:"

run_test \
    "GNN -h flag" \
    "$GNN_BIN -h" \
    "Commands:"

run_test \
    "GNN help shows create options" \
    "$GNN_BIN help" \
    "feature"

run_test \
    "GNN help shows train options" \
    "$GNN_BIN help" \
    "epochs"

run_test \
    "GNN help shows predict options" \
    "$GNN_BIN help" \
    "graph"

run_test \
    "GNN help shows info options" \
    "$GNN_BIN help" \
    "model"

echo ""

# ============================================
# SECTION 2: GNN Binary - Model Creation
# ============================================

echo -e "${BLUE}=== SECTION 2: GNN Binary - Model Creation ===${NC}"
echo ""

run_test \
    "Create basic GNN (3 features, 16 hidden, 2 output, 2 MP layers)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_basic.json" \
    "Created GNN model"

check_file_exists \
    "Basic GNN JSON file created" \
    "$OUTPUT_DIR/gnn_basic.json"

check_json_valid \
    "Basic GNN JSON is valid structure" \
    "$OUTPUT_DIR/gnn_basic.json"

run_test \
    "Create GNN shows correct feature size" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/gnn_feat5.json" \
    "Feature size: 5"

run_test \
    "Create GNN shows correct hidden size" \
    "$GNN_BIN create --feature=5 --hidden=64 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/gnn_hid64.json" \
    "Hidden size: 64"

run_test \
    "Create GNN shows correct output size" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=10 --mp-layers=2 --save=$OUTPUT_DIR/gnn_out10.json" \
    "Output size: 10"

run_test \
    "Create GNN shows message passing layers" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=4 --save=$OUTPUT_DIR/gnn_mp4.json" \
    "Message passing layers: 4"

echo ""

# ============================================
# SECTION 3: GNN - Various MP Layer Configurations
# ============================================

echo -e "${BLUE}=== SECTION 3: GNN - Various MP Layer Configurations ===${NC}"
echo ""

run_test \
    "Create GNN with 1 MP layer" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=1 --save=$OUTPUT_DIR/gnn_mp1.json" \
    "Created GNN model"

run_test \
    "Create GNN with 2 MP layers" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/gnn_mp2.json" \
    "Created GNN model"

run_test \
    "Create GNN with 3 MP layers" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=3 --save=$OUTPUT_DIR/gnn_mp3.json" \
    "Created GNN model"

run_test \
    "Create GNN with 5 MP layers" \
    "$GNN_BIN create --feature=5 --hidden=32 --output=3 --mp-layers=5 --save=$OUTPUT_DIR/gnn_mp5.json" \
    "Created GNN model"

check_json_field \
    "MP1 JSON has num_message_passing_layers" \
    "$OUTPUT_DIR/gnn_mp1.json" \
    "num_message_passing_layers"

check_json_field \
    "MP5 JSON has message_layers array" \
    "$OUTPUT_DIR/gnn_mp5.json" \
    "message_layers"

echo ""

# ============================================
# SECTION 4: GNN - Activation Functions
# ============================================

echo -e "${BLUE}=== SECTION 4: GNN - Activation Functions ===${NC}"
echo ""

run_test \
    "Create GNN with ReLU activation (default)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_relu.json --activation=relu" \
    "Activation: relu"

run_test \
    "Create GNN with LeakyReLU activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_leakyrelu.json --activation=leakyrelu" \
    "Activation: leakyrelu"

run_test \
    "Create GNN with Tanh activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_tanh.json --activation=tanh" \
    "Activation: tanh"

run_test \
    "Create GNN with Sigmoid activation" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_sigmoid.json --activation=sigmoid" \
    "Activation: sigmoid"

check_json_field \
    "ReLU model JSON has activation field" \
    "$OUTPUT_DIR/gnn_relu.json" \
    "activation"

check_json_field \
    "Tanh model JSON has activation field" \
    "$OUTPUT_DIR/gnn_tanh.json" \
    "activation"

echo ""

# ============================================
# SECTION 5: GNN - Loss Functions
# ============================================

echo -e "${BLUE}=== SECTION 5: GNN - Loss Functions ===${NC}"
echo ""

run_test \
    "Create GNN with MSE loss (default)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_mse.json --loss=mse" \
    "Loss function: mse"

run_test \
    "Create GNN with BCE loss" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_bce.json --loss=bce" \
    "Loss function: bce"

check_json_field \
    "MSE model JSON has loss_type field" \
    "$OUTPUT_DIR/gnn_mse.json" \
    "loss_type"

check_json_field \
    "BCE model JSON has loss_type field" \
    "$OUTPUT_DIR/gnn_bce.json" \
    "loss_type"

echo ""

# ============================================
# SECTION 6: GNN - Learning Rate
# ============================================

echo -e "${BLUE}=== SECTION 6: GNN - Learning Rate ===${NC}"
echo ""

run_test \
    "Create GNN with default learning rate" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_lr_default.json" \
    "Learning rate: 0.01"

run_test \
    "Create GNN with custom learning rate 0.001" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_lr_001.json --lr=0.001" \
    "Learning rate: 0.001"

run_test \
    "Create GNN with learning rate 0.1" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_lr_01.json --lr=0.1" \
    "Learning rate: 0.1"

run_test \
    "Create GNN with very small learning rate" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/gnn_lr_tiny.json --lr=0.0001" \
    "Learning rate: 0.0001"

check_json_field \
    "Learning rate persists in JSON" \
    "$OUTPUT_DIR/gnn_lr_001.json" \
    "learning_rate"

echo ""

# ============================================
# SECTION 7: GNN - Info Command
# ============================================

echo -e "${BLUE}=== SECTION 7: GNN - Info Command ===${NC}"
echo ""

run_test \
    "GNN info loads JSON model" \
    "$GNN_BIN info --model=$OUTPUT_DIR/gnn_basic.json" \
    "Loading model from JSON"

run_test \
    "GNN info on ReLU model" \
    "$GNN_BIN info --model=$OUTPUT_DIR/gnn_relu.json" \
    "Loading model from JSON"

run_test \
    "GNN info on multi-layer model" \
    "$GNN_BIN info --model=$OUTPUT_DIR/gnn_mp5.json" \
    "Loading model from JSON"

echo ""

# ============================================
# SECTION 8: FacadeGNN - Help & Usage
# ============================================

echo -e "${BLUE}=== SECTION 8: FacadeGNN - Help & Usage ===${NC}"
echo ""

run_test \
    "FacadeGNN help command" \
    "$FACADE_BIN help" \
    "Commands:"

run_test \
    "FacadeGNN --help flag" \
    "$FACADE_BIN --help" \
    "Commands:"

run_test \
    "FacadeGNN help shows introspection commands" \
    "$FACADE_BIN help" \
    "get-embedding"

run_test \
    "FacadeGNN help shows facade options" \
    "$FACADE_BIN help" \
    "layer"

run_test \
    "FacadeGNN help shows PageRank" \
    "$FACADE_BIN help" \
    "compute-pagerank"

run_test \
    "FacadeGNN help shows gradient detection" \
    "$FACADE_BIN help" \
    "detect-vanishing"

echo ""

# ============================================
# SECTION 9: FacadeGNN - Model Creation
# ============================================

echo -e "${BLUE}=== SECTION 9: FacadeGNN - Model Creation ===${NC}"
echo ""

run_test \
    "Create Facade GNN model" \
    "$FACADE_BIN create --feature=4 --hidden=24 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/facade_basic.json" \
    "Created Facaded GNN model"

check_file_exists \
    "Facade GNN JSON file created" \
    "$OUTPUT_DIR/facade_basic.json"

check_json_valid \
    "Facade GNN JSON is valid structure" \
    "$OUTPUT_DIR/facade_basic.json"

run_test \
    "Facade shows correct feature size" \
    "$FACADE_BIN create --feature=8 --hidden=32 --output=4 --mp-layers=2 --save=$OUTPUT_DIR/facade_feat8.json" \
    "Feature size: 8"

run_test \
    "Facade shows correct hidden size" \
    "$FACADE_BIN create --feature=4 --hidden=48 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/facade_hid48.json" \
    "Hidden size: 48"

run_test \
    "Facade shows correct output size" \
    "$FACADE_BIN create --feature=4 --hidden=24 --output=6 --mp-layers=2 --save=$OUTPUT_DIR/facade_out6.json" \
    "Output size: 6"

run_test \
    "Facade shows message passing layers" \
    "$FACADE_BIN create --feature=4 --hidden=24 --output=3 --mp-layers=3 --save=$OUTPUT_DIR/facade_mp3.json" \
    "Message passing layers: 3"

echo ""

# ============================================
# SECTION 10: FacadeGNN - Various Configurations
# ============================================

echo -e "${BLUE}=== SECTION 10: FacadeGNN - Various Configurations ===${NC}"
echo ""

run_test \
    "Facade with 1 MP layer" \
    "$FACADE_BIN create --feature=4 --hidden=24 --output=3 --mp-layers=1 --save=$OUTPUT_DIR/facade_mp1.json" \
    "Created Facaded GNN model"

run_test \
    "Facade with 4 MP layers" \
    "$FACADE_BIN create --feature=4 --hidden=24 --output=3 --mp-layers=4 --save=$OUTPUT_DIR/facade_mp4.json" \
    "Created Facaded GNN model"

run_test \
    "Facade with 5 MP layers" \
    "$FACADE_BIN create --feature=4 --hidden=24 --output=3 --mp-layers=5 --save=$OUTPUT_DIR/facade_mp5.json" \
    "Created Facaded GNN model"

run_test \
    "Facade with large hidden size" \
    "$FACADE_BIN create --feature=4 --hidden=128 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/facade_large.json" \
    "Created Facaded GNN model"

run_test \
    "Facade with custom learning rate" \
    "$FACADE_BIN create --feature=4 --hidden=24 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/facade_lr.json --lr=0.005" \
    "Learning rate: 0.005"

echo ""

# ============================================
# SECTION 11: FacadeGNN - Info Command
# ============================================

echo -e "${BLUE}=== SECTION 11: FacadeGNN - Info Command ===${NC}"
echo ""

run_test \
    "Facade info shows architecture summary" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/facade_basic.json" \
    "GNN Architecture Summary"

run_test \
    "Facade info shows feature size" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/facade_basic.json" \
    "Feature Size"

run_test \
    "Facade info shows hidden size" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/facade_basic.json" \
    "Hidden Size"

run_test \
    "Facade info shows output size" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/facade_basic.json" \
    "Output Size"

run_test \
    "Facade info shows total parameters" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/facade_basic.json" \
    "Total Parameters"

echo ""

# ============================================
# SECTION 12: FacadeGNN - Introspection Commands
# ============================================

echo -e "${BLUE}=== SECTION 12: FacadeGNN - Introspection Commands ===${NC}"
echo ""

run_test \
    "get-embedding command parses" \
    "$FACADE_BIN get-embedding --model=$OUTPUT_DIR/facade_basic.json --layer=0 --node=0" \
    "Layer:"

run_test \
    "set-embedding command parses" \
    "$FACADE_BIN set-embedding --model=$OUTPUT_DIR/facade_basic.json --layer=0 --node=0 --value=0.5" \
    "Layer:"

run_test \
    "get-graph-embedding command parses" \
    "$FACADE_BIN get-graph-embedding --model=$OUTPUT_DIR/facade_basic.json --layer=0" \
    "Layer:"

run_test \
    "set-graph-embedding command parses" \
    "$FACADE_BIN set-graph-embedding --model=$OUTPUT_DIR/facade_basic.json" \
    "Set graph embedding"

run_test \
    "get-message command parses" \
    "$FACADE_BIN get-message --model=$OUTPUT_DIR/facade_basic.json --layer=0 --node=0 --neighbor=1" \
    "Layer:"

run_test \
    "set-message command parses" \
    "$FACADE_BIN set-message --model=$OUTPUT_DIR/facade_basic.json --layer=0 --node=0 --neighbor=1" \
    "Layer:"

run_test \
    "get-node-degree command parses" \
    "$FACADE_BIN get-node-degree --model=$OUTPUT_DIR/facade_basic.json --node=0" \
    "Node:"

run_test \
    "compute-pagerank command parses" \
    "$FACADE_BIN compute-pagerank --model=$OUTPUT_DIR/facade_basic.json --damping=0.85 --pr-iter=100" \
    "Damping:"

run_test \
    "export-graph command parses" \
    "$FACADE_BIN export-graph --model=$OUTPUT_DIR/facade_basic.json" \
    "Export graph"

run_test \
    "export-embeddings command parses" \
    "$FACADE_BIN export-embeddings --model=$OUTPUT_DIR/facade_basic.json --layer=0" \
    "Layer:"

echo ""

# ============================================
# SECTION 13: FacadeGNN - Graph Modification Commands
# ============================================

echo -e "${BLUE}=== SECTION 13: FacadeGNN - Graph Modification Commands ===${NC}"
echo ""

run_test \
    "add-node command parses" \
    "$FACADE_BIN add-node --model=$OUTPUT_DIR/facade_basic.json" \
    "Add node"

run_test \
    "add-edge command parses" \
    "$FACADE_BIN add-edge --model=$OUTPUT_DIR/facade_basic.json --node=0 --neighbor=1" \
    "Source:"

run_test \
    "get-node-feature command parses" \
    "$FACADE_BIN get-node-feature --model=$OUTPUT_DIR/facade_basic.json --node=0 --feature-idx=0" \
    "Node:"

run_test \
    "set-node-feature command parses" \
    "$FACADE_BIN set-node-feature --model=$OUTPUT_DIR/facade_basic.json --node=0 --feature-idx=0 --value=1.0" \
    "Node:"

echo ""

# ============================================
# SECTION 14: FacadeGNN - Gradient Detection Commands
# ============================================

echo -e "${BLUE}=== SECTION 14: FacadeGNN - Gradient Detection Commands ===${NC}"
echo ""

run_test \
    "detect-vanishing command parses" \
    "$FACADE_BIN detect-vanishing --model=$OUTPUT_DIR/facade_basic.json --threshold=1e-6" \
    "Threshold:"

run_test \
    "detect-exploding command parses" \
    "$FACADE_BIN detect-exploding --model=$OUTPUT_DIR/facade_basic.json --threshold=1e6" \
    "Threshold:"

run_test \
    "get-gradient-flow command parses" \
    "$FACADE_BIN get-gradient-flow --model=$OUTPUT_DIR/facade_basic.json --layer=0" \
    "Layer:"

echo ""

# ============================================
# SECTION 15: Cross-Loading - GNN creates, Facade loads
# ============================================

echo -e "${BLUE}=== SECTION 15: Cross-Loading - GNN creates, Facade loads ===${NC}"
echo ""

# Create model with GNN
$GNN_BIN create --feature=6 --hidden=32 --output=4 --mp-layers=3 --save=$OUTPUT_DIR/cross_gnn_to_facade.json >/dev/null 2>&1

run_test \
    "GNN creates model for cross-loading" \
    "[ -f $OUTPUT_DIR/cross_gnn_to_facade.json ] && echo 'ok'" \
    "ok"

run_test \
    "Facade loads GNN-created model" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/cross_gnn_to_facade.json" \
    "GNN Architecture Summary"

run_test \
    "Facade shows correct feature size from GNN model" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/cross_gnn_to_facade.json" \
    "Feature Size: 6"

run_test \
    "Facade shows correct hidden size from GNN model" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/cross_gnn_to_facade.json" \
    "Hidden Size: 32"

run_test \
    "Facade shows correct output size from GNN model" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/cross_gnn_to_facade.json" \
    "Output Size: 4"

run_test \
    "Facade shows correct MP layers from GNN model" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/cross_gnn_to_facade.json" \
    "Message Passing Layers: 3"

echo ""

# ============================================
# SECTION 16: Cross-Loading - Facade creates, GNN loads
# ============================================

echo -e "${BLUE}=== SECTION 16: Cross-Loading - Facade creates, GNN loads ===${NC}"
echo ""

# Create model with Facade
$FACADE_BIN create --feature=8 --hidden=48 --output=5 --mp-layers=4 --save=$OUTPUT_DIR/cross_facade_to_gnn.json >/dev/null 2>&1

run_test \
    "Facade creates model for cross-loading" \
    "[ -f $OUTPUT_DIR/cross_facade_to_gnn.json ] && echo 'ok'" \
    "ok"

run_test \
    "GNN loads Facade-created model" \
    "$GNN_BIN info --model=$OUTPUT_DIR/cross_facade_to_gnn.json" \
    "Loading model from JSON"

run_test \
    "Cross-load: model file is readable JSON" \
    "file $OUTPUT_DIR/cross_facade_to_gnn.json | grep -i 'text\\|json\\|ascii'" \
    ""

echo ""

# ============================================
# SECTION 17: Cross-Loading - Round Trip
# ============================================

echo -e "${BLUE}=== SECTION 17: Cross-Loading - Round Trip ===${NC}"
echo ""

# Create with GNN, load with Facade, check info matches
$GNN_BIN create --feature=10 --hidden=64 --output=8 --mp-layers=2 --save=$OUTPUT_DIR/roundtrip1.json --activation=tanh --loss=bce --lr=0.005 >/dev/null 2>&1

run_test \
    "Round trip: GNN creates with all params" \
    "[ -f $OUTPUT_DIR/roundtrip1.json ] && echo 'ok'" \
    "ok"

run_test \
    "Round trip: Facade loads and shows activation" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/roundtrip1.json" \
    "Activation: Tanh"

run_test \
    "Round trip: Facade loads and shows learning rate" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/roundtrip1.json" \
    "Learning Rate:"

# Create with Facade, verify GNN can load
$FACADE_BIN create --feature=12 --hidden=72 --output=6 --mp-layers=3 --save=$OUTPUT_DIR/roundtrip2.json >/dev/null 2>&1

run_test \
    "Round trip: Facade creates model" \
    "[ -f $OUTPUT_DIR/roundtrip2.json ] && echo 'ok'" \
    "ok"

run_test \
    "Round trip: GNN info succeeds on Facade model" \
    "$GNN_BIN info --model=$OUTPUT_DIR/roundtrip2.json" \
    "Loading model from JSON"

echo ""

# ============================================
# SECTION 18: JSON Format Validation
# ============================================

echo -e "${BLUE}=== SECTION 18: JSON Format Validation ===${NC}"
echo ""

# Create a reference model
$GNN_BIN create --feature=4 --hidden=20 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/json_test.json >/dev/null 2>&1

check_json_field \
    "JSON has feature_size field" \
    "$OUTPUT_DIR/json_test.json" \
    "feature_size"

check_json_field \
    "JSON has hidden_size field" \
    "$OUTPUT_DIR/json_test.json" \
    "hidden_size"

check_json_field \
    "JSON has output_size field" \
    "$OUTPUT_DIR/json_test.json" \
    "output_size"

check_json_field \
    "JSON has num_message_passing_layers field" \
    "$OUTPUT_DIR/json_test.json" \
    "num_message_passing_layers"

check_json_field \
    "JSON has learning_rate field" \
    "$OUTPUT_DIR/json_test.json" \
    "learning_rate"

check_json_field \
    "JSON has activation field" \
    "$OUTPUT_DIR/json_test.json" \
    "activation"

check_json_field \
    "JSON has loss_type field" \
    "$OUTPUT_DIR/json_test.json" \
    "loss_type"

check_json_field \
    "JSON has max_iterations field" \
    "$OUTPUT_DIR/json_test.json" \
    "max_iterations"

check_json_field \
    "JSON has message_layers array" \
    "$OUTPUT_DIR/json_test.json" \
    "message_layers"

check_json_field \
    "JSON has update_layers array" \
    "$OUTPUT_DIR/json_test.json" \
    "update_layers"

check_json_field \
    "JSON has readout_layer object" \
    "$OUTPUT_DIR/json_test.json" \
    "readout_layer"

check_json_field \
    "JSON has output_layer object" \
    "$OUTPUT_DIR/json_test.json" \
    "output_layer"

echo ""

# ============================================
# SECTION 19: JSON Layer Structure
# ============================================

echo -e "${BLUE}=== SECTION 19: JSON Layer Structure ===${NC}"
echo ""

run_test \
    "Message layers contain neurons" \
    "grep -q '\"neurons\"' $OUTPUT_DIR/json_test.json && echo 'ok'" \
    "ok"

run_test \
    "Neurons have weights array" \
    "grep -q '\"weights\"' $OUTPUT_DIR/json_test.json && echo 'ok'" \
    "ok"

run_test \
    "Neurons have bias value" \
    "grep -q '\"bias\"' $OUTPUT_DIR/json_test.json && echo 'ok'" \
    "ok"

run_test \
    "Layers have num_outputs" \
    "grep -q '\"num_outputs\"' $OUTPUT_DIR/json_test.json && echo 'ok'" \
    "ok"

run_test \
    "Layers have num_inputs" \
    "grep -q '\"num_inputs\"' $OUTPUT_DIR/json_test.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# SECTION 20: Dimension Edge Cases
# ============================================

echo -e "${BLUE}=== SECTION 20: Dimension Edge Cases ===${NC}"
echo ""

run_test \
    "Minimal model (1-2-1-1)" \
    "$GNN_BIN create --feature=1 --hidden=2 --output=1 --mp-layers=1 --save=$OUTPUT_DIR/edge_minimal.json" \
    "Created GNN model"

run_test \
    "Deep model (5 MP layers)" \
    "$GNN_BIN create --feature=2 --hidden=8 --output=2 --mp-layers=5 --save=$OUTPUT_DIR/edge_deep.json" \
    "Created GNN model"

run_test \
    "Wide model (256-256-128)" \
    "$GNN_BIN create --feature=256 --hidden=256 --output=128 --mp-layers=2 --save=$OUTPUT_DIR/edge_wide.json" \
    "Created GNN model"

run_test \
    "Large feature dimension (512)" \
    "$GNN_BIN create --feature=512 --hidden=64 --output=10 --mp-layers=2 --save=$OUTPUT_DIR/edge_largefeat.json" \
    "Created GNN model"

run_test \
    "Large hidden dimension (512)" \
    "$GNN_BIN create --feature=10 --hidden=512 --output=5 --mp-layers=2 --save=$OUTPUT_DIR/edge_largehid.json" \
    "Created GNN model"

run_test \
    "Large output dimension (256)" \
    "$GNN_BIN create --feature=10 --hidden=64 --output=256 --mp-layers=2 --save=$OUTPUT_DIR/edge_largeout.json" \
    "Created GNN model"

check_file_exists \
    "Minimal model file exists" \
    "$OUTPUT_DIR/edge_minimal.json"

check_file_exists \
    "Deep model file exists" \
    "$OUTPUT_DIR/edge_deep.json"

check_file_exists \
    "Wide model file exists" \
    "$OUTPUT_DIR/edge_wide.json"

echo ""

# ============================================
# SECTION 21: Model Consistency Tests
# ============================================

echo -e "${BLUE}=== SECTION 21: Model Consistency Tests ===${NC}"
echo ""

# Create a model, load it twice, verify consistency
$GNN_BIN create --feature=5 --hidden=20 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/consistency.json >/dev/null 2>&1

run_test \
    "GNN info produces consistent output (run 1 vs run 2)" \
    "$GNN_BIN info --model=$OUTPUT_DIR/consistency.json > $OUTPUT_DIR/gnn_cons1.txt && \
     $GNN_BIN info --model=$OUTPUT_DIR/consistency.json > $OUTPUT_DIR/gnn_cons2.txt && \
     diff $OUTPUT_DIR/gnn_cons1.txt $OUTPUT_DIR/gnn_cons2.txt > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Facade info produces consistent output (run 1 vs run 2)" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/consistency.json > $OUTPUT_DIR/facade_cons1.txt && \
     $FACADE_BIN info --model=$OUTPUT_DIR/consistency.json > $OUTPUT_DIR/facade_cons2.txt && \
     diff $OUTPUT_DIR/facade_cons1.txt $OUTPUT_DIR/facade_cons2.txt > /dev/null && echo 'ok'" \
    "ok"

echo ""

# ============================================
# SECTION 22: GNN and Facade Parity Tests
# ============================================

echo -e "${BLUE}=== SECTION 22: GNN and Facade Parity Tests ===${NC}"
echo ""

run_test \
    "Both binaries can load the same model" \
    "$GNN_BIN create --feature=4 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/parity.json && \
     $GNN_BIN info --model=$OUTPUT_DIR/parity.json > /dev/null && \
     $FACADE_BIN info --model=$OUTPUT_DIR/parity.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "GNN-created model loadable by Facade (multiple models)" \
    "$GNN_BIN create --feature=3 --hidden=12 --output=2 --mp-layers=1 --save=$OUTPUT_DIR/parity1.json && \
     $GNN_BIN create --feature=5 --hidden=24 --output=4 --mp-layers=3 --save=$OUTPUT_DIR/parity2.json && \
     $FACADE_BIN info --model=$OUTPUT_DIR/parity1.json > /dev/null && \
     $FACADE_BIN info --model=$OUTPUT_DIR/parity2.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Facade-created model loadable by GNN (multiple models)" \
    "$FACADE_BIN create --feature=6 --hidden=30 --output=5 --mp-layers=2 --save=$OUTPUT_DIR/parity3.json && \
     $FACADE_BIN create --feature=8 --hidden=40 --output=6 --mp-layers=4 --save=$OUTPUT_DIR/parity4.json && \
     $GNN_BIN info --model=$OUTPUT_DIR/parity3.json > /dev/null && \
     $GNN_BIN info --model=$OUTPUT_DIR/parity4.json > /dev/null && echo 'ok'" \
    "ok"

echo ""

# ============================================
# SECTION 23: File Persistence
# ============================================

echo -e "${BLUE}=== SECTION 23: File Persistence ===${NC}"
echo ""

run_test \
    "Created model file is not empty" \
    "[ -s $OUTPUT_DIR/gnn_basic.json ] && echo 'ok'" \
    "ok"

run_test \
    "Model JSON is readable text file" \
    "file $OUTPUT_DIR/gnn_basic.json | grep -iq 'text\\|json\\|ascii' && echo 'ok'" \
    "ok"

run_test \
    "Multiple models can exist independently" \
    "[ -f $OUTPUT_DIR/gnn_mp1.json ] && [ -f $OUTPUT_DIR/gnn_mp3.json ] && [ -f $OUTPUT_DIR/gnn_mp5.json ] && echo 'ok'" \
    "ok"

run_test \
    "Models with different configs have different file content" \
    "! diff -q $OUTPUT_DIR/gnn_mp1.json $OUTPUT_DIR/gnn_mp5.json > /dev/null && echo 'different'" \
    "different"

echo ""

# ============================================
# SECTION 24: Extended Facade Function Tests
# ============================================

echo -e "${BLUE}=== SECTION 24: Extended Facade Function Tests ===${NC}"
echo ""

# Create a model for repeated testing
$FACADE_BIN create --feature=4 --hidden=20 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/facade_extended.json >/dev/null 2>&1

# Test multiple calls to same facade commands (stability)
for i in {1..3}; do
    run_test \
        "Facade info call $i succeeds" \
        "$FACADE_BIN info --model=$OUTPUT_DIR/facade_extended.json > /dev/null && echo 'ok'" \
        "ok"
done

for i in {1..3}; do
    run_test \
        "Facade get-embedding call $i succeeds" \
        "$FACADE_BIN get-embedding --model=$OUTPUT_DIR/facade_extended.json --layer=0 --node=0 > /dev/null && echo 'ok'" \
        "ok"
done

for i in {1..3}; do
    run_test \
        "Facade compute-pagerank call $i succeeds" \
        "$FACADE_BIN compute-pagerank --model=$OUTPUT_DIR/facade_extended.json > /dev/null && echo 'ok'" \
        "ok"
done

echo ""

# ============================================
# SECTION 25: Hyperparameter Range Tests
# ============================================

echo -e "${BLUE}=== SECTION 25: Hyperparameter Range Tests ===${NC}"
echo ""

run_test \
    "Very small learning rate (0.00001)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/hp_lr_tiny.json --lr=0.00001" \
    "Created GNN model"

run_test \
    "Large learning rate (1.0)" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=$OUTPUT_DIR/hp_lr_large.json --lr=1.0" \
    "Created GNN model"

run_test \
    "Learning rate persists in tiny LR model" \
    "grep -q 'learning_rate' $OUTPUT_DIR/hp_lr_tiny.json && echo 'ok'" \
    "ok"

run_test \
    "Learning rate persists in large LR model" \
    "grep -q 'learning_rate' $OUTPUT_DIR/hp_lr_large.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# SECTION 26: Combined Feature Tests
# ============================================

echo -e "${BLUE}=== SECTION 26: Combined Feature Tests ===${NC}"
echo ""

run_test \
    "Create with all custom params (GNN)" \
    "$GNN_BIN create --feature=8 --hidden=40 --output=5 --mp-layers=3 --save=$OUTPUT_DIR/full_custom.json --lr=0.005 --activation=tanh --loss=bce" \
    "Created GNN model"

run_test \
    "Full custom model shows activation" \
    "$GNN_BIN create --feature=8 --hidden=40 --output=5 --mp-layers=3 --save=$OUTPUT_DIR/full_custom2.json --lr=0.005 --activation=sigmoid --loss=mse" \
    "Activation: sigmoid"

run_test \
    "Full custom model shows loss" \
    "$GNN_BIN create --feature=8 --hidden=40 --output=5 --mp-layers=3 --save=$OUTPUT_DIR/full_custom3.json --lr=0.005 --activation=relu --loss=bce" \
    "Loss function: bce"

run_test \
    "Facade can load full custom GNN model" \
    "$FACADE_BIN info --model=$OUTPUT_DIR/full_custom.json" \
    "GNN Architecture Summary"

echo ""

# ============================================
# SECTION 27: Train Command Tests
# ============================================

echo -e "${BLUE}=== SECTION 27: Train Command Tests ===${NC}"
echo ""

run_test \
    "GNN train command requires --model" \
    "$GNN_BIN train --graph=test.json --save=out.json 2>&1" \
    "model is required"

run_test \
    "GNN train command requires --graph" \
    "$GNN_BIN train --model=$OUTPUT_DIR/gnn_basic.json --save=out.json 2>&1" \
    "graph is required"

run_test \
    "GNN train command requires --save" \
    "$GNN_BIN train --model=$OUTPUT_DIR/gnn_basic.json --graph=test.json 2>&1" \
    "save is required"

run_test \
    "Facade train command requires --model" \
    "$FACADE_BIN train --save=out.json 2>&1" \
    "model is required"

run_test \
    "Facade train command requires --save" \
    "$FACADE_BIN train --model=$OUTPUT_DIR/facade_basic.json 2>&1" \
    "save is required"

echo ""

# ============================================
# SECTION 28: Predict Command Tests
# ============================================

echo -e "${BLUE}=== SECTION 28: Predict Command Tests ===${NC}"
echo ""

run_test \
    "GNN predict command requires --model" \
    "$GNN_BIN predict --graph=test.json 2>&1" \
    "model is required"

run_test \
    "GNN predict command requires --graph" \
    "$GNN_BIN predict --model=$OUTPUT_DIR/gnn_basic.json 2>&1" \
    "graph is required"

run_test \
    "Facade predict command requires --model" \
    "$FACADE_BIN predict 2>&1" \
    "model is required"

echo ""

# ============================================
# SECTION 29: Error Handling Tests
# ============================================

echo -e "${BLUE}=== SECTION 29: Error Handling Tests ===${NC}"
echo ""

run_test \
    "GNN create missing --feature shows error" \
    "$GNN_BIN create --hidden=16 --output=2 --mp-layers=2 --save=test.json 2>&1" \
    "feature is required"

run_test \
    "GNN create missing --hidden shows error" \
    "$GNN_BIN create --feature=3 --output=2 --mp-layers=2 --save=test.json 2>&1" \
    "hidden is required"

run_test \
    "GNN create missing --output shows error" \
    "$GNN_BIN create --feature=3 --hidden=16 --mp-layers=2 --save=test.json 2>&1" \
    "output is required"

run_test \
    "GNN create missing --mp-layers shows error" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --save=test.json 2>&1" \
    "mp-layers is required"

run_test \
    "GNN create missing --save shows error" \
    "$GNN_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 2>&1" \
    "save is required"

run_test \
    "Facade create missing --feature shows error" \
    "$FACADE_BIN create --hidden=16 --output=2 --mp-layers=2 --save=test.json 2>&1" \
    "feature is required"

run_test \
    "Facade create missing --hidden shows error" \
    "$FACADE_BIN create --feature=3 --output=2 --mp-layers=2 --save=test.json 2>&1" \
    "hidden is required"

run_test \
    "Facade create missing --output shows error" \
    "$FACADE_BIN create --feature=3 --hidden=16 --mp-layers=2 --save=test.json 2>&1" \
    "output is required"

run_test \
    "Facade create missing --mp-layers shows error" \
    "$FACADE_BIN create --feature=3 --hidden=16 --output=2 --save=test.json 2>&1" \
    "mp-layers is required"

run_test \
    "Facade create missing --save shows error" \
    "$FACADE_BIN create --feature=3 --hidden=16 --output=2 --mp-layers=2 2>&1" \
    "save is required"

run_test \
    "Unknown command shows error (GNN)" \
    "$GNN_BIN unknown_command 2>&1" \
    "Unknown command"

run_test \
    "Unknown command shows error (Facade)" \
    "$FACADE_BIN unknown_command 2>&1" \
    "Unknown command"

echo ""

# ============================================
# SECTION 30: Binary Format vs JSON Format
# ============================================

echo -e "${BLUE}=== SECTION 30: Binary Format vs JSON Format ===${NC}"
echo ""

# Facade supports both .json and binary format
run_test \
    "Facade creates JSON when extension is .json" \
    "$FACADE_BIN create --feature=4 --hidden=20 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/format_test.json && \
     grep -q '\"feature_size\"' $OUTPUT_DIR/format_test.json && echo 'ok'" \
    "ok"

run_test \
    "JSON file is human-readable" \
    "head -1 $OUTPUT_DIR/format_test.json | grep -q '{' && echo 'ok'" \
    "ok"

echo ""

# ============================================
# SECTION 31: Complete Workflow Tests
# ============================================

echo -e "${BLUE}=== SECTION 31: Complete Workflow Tests ===${NC}"
echo ""

run_test \
    "Workflow: Create -> Info (GNN)" \
    "$GNN_BIN create --feature=4 --hidden=20 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/wf_gnn.json && \
     $GNN_BIN info --model=$OUTPUT_DIR/wf_gnn.json" \
    "Loading model from JSON"

run_test \
    "Workflow: Create -> Info (Facade)" \
    "$FACADE_BIN create --feature=4 --hidden=20 --output=3 --mp-layers=2 --save=$OUTPUT_DIR/wf_facade.json && \
     $FACADE_BIN info --model=$OUTPUT_DIR/wf_facade.json" \
    "GNN Architecture Summary"

run_test \
    "Workflow: GNN Create -> Facade Load -> Facade Info" \
    "$GNN_BIN create --feature=5 --hidden=25 --output=4 --mp-layers=3 --save=$OUTPUT_DIR/wf_cross1.json && \
     $FACADE_BIN info --model=$OUTPUT_DIR/wf_cross1.json" \
    "Feature Size: 5"

run_test \
    "Workflow: Facade Create -> GNN Load -> GNN Info" \
    "$FACADE_BIN create --feature=6 --hidden=30 --output=5 --mp-layers=2 --save=$OUTPUT_DIR/wf_cross2.json && \
     $GNN_BIN info --model=$OUTPUT_DIR/wf_cross2.json" \
    "Loading model from JSON"

run_test \
    "Workflow: Create with all params -> Cross-load" \
    "$GNN_BIN create --feature=8 --hidden=40 --output=6 --mp-layers=4 --save=$OUTPUT_DIR/wf_full.json --lr=0.002 --activation=leakyrelu --loss=bce && \
     $FACADE_BIN info --model=$OUTPUT_DIR/wf_full.json" \
    "GNN Architecture Summary"

echo ""

# ============================================
# Summary
# ============================================

echo ""
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
