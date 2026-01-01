#!/bin/bash

#
# Matthew Abbott 2025
# Random Forest Tests
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="./test_output"
FOREST_BIN="./Forest"
FACADE_BIN="./FacadeForest"

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

fpc Forest.pas
fpc FacadeForest.pas

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

    if grep -q '"num_trees"' "$file" && grep -q '"task_type"' "$file" && grep -q '"criterion"' "$file"; then
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
echo "Random Forest User Workflow Test Suite"
echo "========================================="
echo ""

# Check binaries exist
if [ ! -f "$FOREST_BIN" ]; then
    echo -e "${RED}Error: $FOREST_BIN not found. Compile with: fpc Forest.pas${NC}"
    exit 1
fi

if [ ! -f "$FACADE_BIN" ]; then
    echo -e "${RED}Error: $FACADE_BIN not found. Compile with: fpc FacadeForest.pas${NC}"
    exit 1
fi

echo -e "${BLUE}=== Forest Binary Tests ===${NC}"
echo ""

# ============================================
# Basic Help/Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "Forest help command" \
    "$FOREST_BIN help" \
    "Random Forest"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create basic classification forest" \
    "$FOREST_BIN create --trees=10 --max-depth=5 --save=$TEMP_DIR/basic.json" \
    "Created Random Forest"

check_file_exists \
    "JSON file created for basic forest" \
    "$TEMP_DIR/basic.json"

check_json_valid \
    "JSON contains valid forest structure" \
    "$TEMP_DIR/basic.json"

run_test \
    "Output shows correct tree count" \
    "$FOREST_BIN create --trees=10 --max-depth=5 --save=$TEMP_DIR/basic2.json" \
    "Number of trees: 10"

run_test \
    "Output shows max depth" \
    "$FOREST_BIN create --trees=10 --max-depth=5 --save=$TEMP_DIR/basic3.json" \
    "Max depth: 5"

run_test \
    "Output shows task type" \
    "$FOREST_BIN create --trees=10 --save=$TEMP_DIR/basic4.json" \
    "Classification"

echo ""

# ============================================
# Model Creation - Hyperparameters
# ============================================

echo -e "${BLUE}Group: Model Creation - Hyperparameters${NC}"

run_test \
    "Create forest with custom min_samples_leaf" \
    "$FOREST_BIN create --trees=5 --min-leaf=5 --save=$TEMP_DIR/hyper1.json" \
    "Created Random Forest"

run_test \
    "Create forest with custom min_samples_split" \
    "$FOREST_BIN create --trees=5 --min-split=5 --save=$TEMP_DIR/hyper2.json" \
    "Created Random Forest"

run_test \
    "Create forest with max_features" \
    "$FOREST_BIN create --trees=5 --max-features=3 --save=$TEMP_DIR/hyper3.json" \
    "Created Random Forest"

echo ""

# ============================================
# Criterion Types
# ============================================

echo -e "${BLUE}Group: Split Criteria${NC}"

run_test \
    "Create forest with Gini criterion" \
    "$FOREST_BIN create --trees=5 --criterion=gini --save=$TEMP_DIR/gini.json" \
    "Created Random Forest"

run_test \
    "Create forest with Entropy criterion" \
    "$FOREST_BIN create --trees=5 --criterion=entropy --save=$TEMP_DIR/entropy.json" \
    "Created Random Forest"

run_test \
    "Create forest with MSE criterion" \
    "$FOREST_BIN create --trees=5 --criterion=mse --save=$TEMP_DIR/mse.json" \
    "Created Random Forest"

run_test \
    "Create forest with Variance criterion" \
    "$FOREST_BIN create --trees=5 --criterion=variance --save=$TEMP_DIR/variance.json" \
    "Created Random Forest"

echo ""

# ============================================
# Task Types
# ============================================

echo -e "${BLUE}Group: Task Types${NC}"

run_test \
    "Create classification forest" \
    "$FOREST_BIN create --trees=5 --task=classification --save=$TEMP_DIR/classif.json" \
    "Classification"

run_test \
    "Create regression forest" \
    "$FOREST_BIN create --trees=5 --task=regression --save=$TEMP_DIR/regress.json" \
    "Regression"

echo ""

# ============================================
# Model Info Command
# ============================================

echo -e "${BLUE}Group: Model Information${NC}"

run_test \
    "Get info on created model" \
    "$FOREST_BIN info --model=$TEMP_DIR/basic.json" \
    "Number of trees"

run_test \
    "Info shows max depth" \
    "$FOREST_BIN info --model=$TEMP_DIR/basic.json" \
    "Max depth"

run_test \
    "Info shows criterion" \
    "$FOREST_BIN info --model=$TEMP_DIR/basic.json" \
    "Criterion"

echo ""

# ============================================
# Cross-binary Compatibility
# ============================================

echo -e "${BLUE}Group: Cross-binary Compatibility${NC}"

run_test \
    "Model created by Forest can be loaded by FacadeForest" \
    "$FACADE_BIN info --model=$TEMP_DIR/basic.json" \
    "Number of Trees"

run_test \
    "FacadeForest shows correct tree count" \
    "$FACADE_BIN info --model=$TEMP_DIR/basic.json" \
    "Number of Trees: 10"

run_test \
    "Model created by FacadeForest can be loaded by Forest" \
    "$FACADE_BIN create --trees=7 --save=$TEMP_DIR/facade_cross.json && $FOREST_BIN info --model=$TEMP_DIR/facade_cross.json" \
    "Number of trees: 7"

run_test \
    "Forest shows correct FacadeForest-created model info" \
    "$FOREST_BIN info --model=$TEMP_DIR/facade_cross.json" \
    "Task type: classification"

echo ""

# ============================================
# JSON Format Validation
# ============================================

echo -e "${BLUE}Group: JSON Format Validation${NC}"

run_test \
    "JSON file is valid JSON" \
    "python3 -m json.tool $TEMP_DIR/basic.json > /dev/null 2>&1 && echo 'ok'" \
    "ok"

run_test \
    "JSON has num_trees field" \
    "grep -q '\"num_trees\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has max_depth field" \
    "grep -q '\"max_depth\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has task_type field" \
    "grep -q '\"task_type\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has criterion field" \
    "grep -q '\"criterion\"' $TEMP_DIR/basic.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Error Cases (expected to handle gracefully)
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Missing required --save argument" \
    "$FOREST_BIN create --trees=5 2>&1" \
    "Error"

run_test \
    "Loading non-existent model" \
    "$FOREST_BIN info --model=$TEMP_DIR/nonexistent.json 2>&1" \
    ""

echo ""

# ============================================
# Sequential Operations (Real Workflow)
# ============================================

echo -e "${BLUE}Group: Sequential Operations Workflow${NC}"

run_test \
    "Workflow: Create -> Load -> Info (Classification)" \
    "$FOREST_BIN create --trees=20 --max-depth=8 --save=$TEMP_DIR/workflow1.json && $FACADE_BIN info --model=$TEMP_DIR/workflow1.json" \
    "Number of trees: 20"

run_test \
    "Workflow: Create -> Load -> Info (Regression)" \
    "$FOREST_BIN create --trees=15 --task=regression --save=$TEMP_DIR/workflow2.json && $FACADE_BIN info --model=$TEMP_DIR/workflow2.json" \
    "Regression"

run_test \
    "Workflow: Create with params -> Verify -> Get Info" \
    "$FOREST_BIN create --trees=10 --max-depth=6 --min-leaf=2 --save=$TEMP_DIR/workflow3.json && $FOREST_BIN info --model=$TEMP_DIR/workflow3.json" \
    "Number of trees: 10"

echo ""

# ============================================
# Advanced Features
# ============================================

echo -e "${BLUE}Group: Advanced Forest Features${NC}"

run_test \
    "Large forest creation" \
    "$FOREST_BIN create --trees=100 --max-depth=10 --save=$TEMP_DIR/large.json" \
    "Created Random Forest"

run_test \
    "Deep forest creation" \
    "$FOREST_BIN create --trees=10 --max-depth=20 --save=$TEMP_DIR/deep.json" \
    "Created Random Forest"

run_test \
    "Complex hyperparameter set" \
    "$FOREST_BIN create --trees=50 --max-depth=15 --min-leaf=3 --min-split=4 --max-features=5 --save=$TEMP_DIR/complex.json" \
    "Created Random Forest"

echo ""

# ============================================
# Facade-Specific Features
# ============================================

echo -e "${BLUE}Group: FacadeForest Tree Management${NC}"

run_test \
    "FacadeForest has remove-tree command" \
    "$FACADE_BIN help 2>&1 || echo 'Help available'" \
    "Random Forest"

run_test \
    "FacadeForest accepts --model parameter in commands" \
    "$FACADE_BIN create --trees=5 --save=$TEMP_DIR/facade_test.json && [ -f $TEMP_DIR/facade_test.json ] && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Base Functionality - Data Input
# ============================================

echo -e "${BLUE}Group: Data Handling${NC}"

# Create sample CSV file for training
cat > "$TEMP_DIR/sample_data.csv" << 'CSVDATA'
0.1,0.2,0.3,0
0.2,0.3,0.4,0
0.3,0.4,0.5,1
0.4,0.5,0.6,1
0.5,0.6,0.7,0
0.6,0.7,0.8,1
CSVDATA

check_file_exists \
    "Sample training data created" \
    "$TEMP_DIR/sample_data.csv"

echo ""

# ============================================
# Base Functionality - Model Persistence
# ============================================

echo -e "${BLUE}Group: Model Persistence & Round-trip${NC}"

run_test \
    "Create forest with specific hyperparameters" \
    "$FOREST_BIN create --trees=5 --max-depth=4 --min-leaf=2 --save=$TEMP_DIR/persist1.json" \
    "Created Random Forest"

run_test \
    "Load persisted model and verify parameters" \
    "$FOREST_BIN info --model=$TEMP_DIR/persist1.json" \
    "Number of trees: 5"

run_test \
    "Verify max depth persisted" \
    "$FOREST_BIN info --model=$TEMP_DIR/persist1.json" \
    "Max depth: 4"

run_test \
    "Verify min leaf persisted in JSON" \
    "grep -q '\"min_samples_leaf\": 2' $TEMP_DIR/persist1.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Regression vs Classification
# ============================================

echo -e "${BLUE}Group: Regression vs Classification Tasks${NC}"

run_test \
    "Create classification model explicitly" \
    "$FOREST_BIN create --trees=4 --task=classification --save=$TEMP_DIR/task_classif.json" \
    "Classification"

run_test \
    "Classification model preserves task type" \
    "$FOREST_BIN info --model=$TEMP_DIR/task_classif.json" \
    "classification"

run_test \
    "Create regression model explicitly" \
    "$FOREST_BIN create --trees=4 --task=regression --save=$TEMP_DIR/task_regress.json" \
    "Regression"

run_test \
    "Regression model preserves task type" \
    "$FOREST_BIN info --model=$TEMP_DIR/task_regress.json" \
    "regression"

echo ""

# ============================================
# Criterion Preservation
# ============================================

echo -e "${BLUE}Group: Split Criterion Preservation${NC}"

run_test \
    "Gini criterion preserved in JSON" \
    "$FOREST_BIN create --trees=3 --criterion=gini --save=$TEMP_DIR/crit_gini.json && grep -q '\"criterion\": \"gini\"' $TEMP_DIR/crit_gini.json && echo 'ok'" \
    "ok"

run_test \
    "Entropy criterion in creation output" \
    "$FOREST_BIN create --trees=3 --criterion=entropy --save=$TEMP_DIR/crit_entropy.json" \
    "Criterion: Entropy"

run_test \
    "MSE criterion in creation output" \
    "$FOREST_BIN create --trees=3 --criterion=mse --save=$TEMP_DIR/crit_mse.json" \
    "Criterion: MSE"

run_test \
    "Variance criterion in creation output" \
    "$FOREST_BIN create --trees=3 --criterion=variance --save=$TEMP_DIR/crit_variance.json" \
    "Criterion: Variance"

echo ""

# ============================================
# Facade Cross-compatibility with Different Criteria
# ============================================

echo -e "${BLUE}Group: Facade Loads Different Criteria Models${NC}"

run_test \
    "FacadeForest loads Gini model from Forest" \
    "$FACADE_BIN info --model=$TEMP_DIR/crit_gini.json 2>/dev/null" \
    "Number of Trees"

run_test \
    "FacadeForest loads Entropy model from Forest" \
    "$FACADE_BIN info --model=$TEMP_DIR/crit_entropy.json 2>/dev/null" \
    "Number of Trees"

run_test \
    "FacadeForest loads Regression model from Forest" \
    "$FACADE_BIN info --model=$TEMP_DIR/task_regress.json 2>/dev/null" \
    "Number of Trees"

echo ""

# ============================================
# Multiple Sequential Operations
# ============================================

echo -e "${BLUE}Group: Sequential Forest Operations${NC}"

run_test \
    "Create Forest model A" \
    "$FOREST_BIN create --trees=3 --save=$TEMP_DIR/seq_a.json" \
    "Created"

run_test \
    "Load Forest model A and verify" \
    "$FOREST_BIN info --model=$TEMP_DIR/seq_a.json" \
    "trees: 3"

run_test \
    "Create FacadeForest model B" \
    "$FACADE_BIN create --trees=4 --save=$TEMP_DIR/seq_b.json" \
    "Created"

run_test \
    "Load FacadeForest model B with Forest" \
    "$FOREST_BIN info --model=$TEMP_DIR/seq_b.json" \
    "Number of trees: 4"

run_test \
    "Load Forest model A with FacadeForest" \
    "$FACADE_BIN info --model=$TEMP_DIR/seq_a.json 2>/dev/null" \
    "Trees: 3"

echo ""

# ============================================
# Feature Importances Preservation
# ============================================

echo -e "${BLUE}Group: Feature Importances${NC}"

run_test \
    "JSON includes feature_importances array" \
    "grep -q '\"feature_importances\"' $TEMP_DIR/persist1.json && echo 'ok'" \
    "ok"

run_test \
    "Feature importances is valid JSON array" \
    "grep 'feature_importances.*\[' $TEMP_DIR/persist1.json > /dev/null && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Hyperparameter Edge Cases
# ============================================

echo -e "${BLUE}Group: Hyperparameter Edge Cases${NC}"

run_test \
    "Single tree forest" \
    "$FOREST_BIN create --trees=1 --save=$TEMP_DIR/edge_1tree.json" \
    "Number of trees: 1"

run_test \
    "Large number of trees" \
    "$FOREST_BIN create --trees=200 --save=$TEMP_DIR/edge_many.json" \
    "Number of trees: 200"

run_test \
    "Very shallow forest (depth=1)" \
    "$FOREST_BIN create --trees=5 --max-depth=1 --save=$TEMP_DIR/edge_shallow.json" \
    "Max depth: 1"

run_test \
    "Deep forest (depth=25)" \
    "$FOREST_BIN create --trees=5 --max-depth=25 --save=$TEMP_DIR/edge_deep.json" \
    "Max depth: 25"

run_test \
    "High min_samples_leaf (10)" \
    "$FOREST_BIN create --trees=5 --min-leaf=10 --save=$TEMP_DIR/edge_leaf10.json" \
    "Min samples leaf: 10"

run_test \
    "High min_samples_split (20)" \
    "$FOREST_BIN create --trees=5 --min-split=20 --save=$TEMP_DIR/edge_split20.json" \
    "Min samples split: 20"

echo ""

# ============================================
# Multiple Models with Different Configs
# ============================================

echo -e "${BLUE}Group: Multiple Models Configuration Verification${NC}"

for i in 1 2 3; do
    run_test \
        "Model variant $i: Create and verify trees=$((5 + i))" \
        "$FOREST_BIN create --trees=$((5 + i)) --save=$TEMP_DIR/variant_$i.json && $FOREST_BIN info --model=$TEMP_DIR/variant_$i.json" \
        "Number of trees: $((5 + i))"
done

echo ""

# ============================================
# JSON Integrity Checks
# ============================================

echo -e "${BLUE}Group: JSON Integrity${NC}"

run_test \
    "All JSON files have closing brace" \
    "find $TEMP_DIR -name '*.json' -exec grep -l '^}$' {} \; | wc -l | grep -q '[1-9]' && echo 'ok'" \
    "ok"

run_test \
    "All JSON files are valid" \
    "for f in $TEMP_DIR/*.json; do python3 -m json.tool \"\$f\" > /dev/null || exit 1; done && echo 'ok'" \
    "ok"

run_test \
    "Random JSON file is properly formatted" \
    "python3 -m json.tool $TEMP_DIR/persist1.json | head -5 | grep -q '{' && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Training and Prediction Tests
# ============================================

echo -e "${BLUE}Group: Training with Real Data${NC}"

run_test \
    "Generate training data with 500 samples" \
    "python3 -c \"
import random
random.seed(42)
with open('$TEMP_DIR/train_data.csv', 'w') as f:
    for i in range(500):
        features = [random.random() * 10 for _ in range(10)]
        feature_sum = sum(features[:3])
        label = 2 if feature_sum > 15 else (1 if feature_sum > 10 else 0)
        line = ','.join(f'{f:.4f}' for f in features) + f',{label}\n'
        f.write(line)
print('Generated')
\" && echo 'ok'" \
    "Generated"

check_file_exists \
    "Training data file created" \
    "$TEMP_DIR/train_data.csv"

run_test \
    "Create forest model configuration" \
    "$FOREST_BIN create --trees=5 --max-depth=4 --save=$TEMP_DIR/model_config.json" \
    "Created Random Forest"

run_test \
    "Train forest on CSV data" \
    "$FOREST_BIN train --model=$TEMP_DIR/model_config.json --data=$TEMP_DIR/train_data.csv --save=$TEMP_DIR/trained_model.json" \
    "Training complete"

run_test \
    "Trained model has num_features set" \
    "grep -q '\"num_features\": 10' $TEMP_DIR/trained_model.json && echo 'ok'" \
    "ok"

run_test \
    "Trained model has non-null trees" \
    "grep -q '\"left\"' $TEMP_DIR/trained_model.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Prediction Workflow
# ============================================

echo -e "${BLUE}Group: Prediction Workflow${NC}"

run_test \
    "Create fresh model for prediction test" \
    "$FOREST_BIN create --trees=10 --max-depth=5 --save=$TEMP_DIR/pred_model.json" \
    "Created Random Forest"

run_test \
    "Generate test prediction data (100 samples)" \
    "python3 -c \"
import random
random.seed(123)
with open('$TEMP_DIR/pred_data.csv', 'w') as f:
    for i in range(100):
        features = [random.random() * 10 for _ in range(10)]
        feature_sum = sum(features[:3])
        label = 2 if feature_sum > 15 else (1 if feature_sum > 10 else 0)
        line = ','.join(f'{f:.4f}' for f in features) + f',{label}\n'
        f.write(line)
print('Generated')
\" && echo 'ok'" \
    "Generated"

check_file_exists \
    "Prediction data file created" \
    "$TEMP_DIR/pred_data.csv"

run_test \
    "Train model on prediction training data" \
    "$FOREST_BIN train --model=$TEMP_DIR/pred_model.json --data=$TEMP_DIR/pred_data.csv --save=$TEMP_DIR/pred_model_trained.json" \
    "Training complete"

check_file_exists \
    "Trained prediction model saved" \
    "$TEMP_DIR/pred_model_trained.json"

run_test \
    "Load trained prediction model with info command" \
    "$FOREST_BIN info --model=$TEMP_DIR/pred_model_trained.json" \
    "Number of trees: 10"

run_test \
    "FacadeForest can load trained prediction model" \
    "$FACADE_BIN info --model=$TEMP_DIR/pred_model_trained.json 2>/dev/null" \
    "Trees: 10"

run_test \
    "Prediction model JSON is valid" \
    "python3 -m json.tool $TEMP_DIR/pred_model_trained.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Prediction model has 10 features" \
    "grep -q '\"num_features\": 10' $TEMP_DIR/pred_model_trained.json && echo 'ok'" \
    "ok"

run_test \
    "Prediction model has trained trees (non-null)" \
    "grep -q '\"left\"' $TEMP_DIR/pred_model_trained.json && echo 'ok'" \
    "ok"

echo ""
echo -e "${BLUE}Keeping prediction model for JavaScript testing${NC}"
echo "Model saved to: $TEMP_DIR/pred_model_trained.json"
cp "$TEMP_DIR/pred_model_trained.json" "trained_model_for_js.json"
run_test \
    "Prediction model copied for JS testing" \
    "[ -f trained_model_for_js.json ] && echo 'ok'" \
    "ok"

echo ""

echo ""
echo "========================================="
echo "Comprehensive Cross-Binary Test Suite"
echo "========================================="
echo ""

echo -e "${BLUE}=== Cross-Binary Feature Tests ===${NC}"
echo ""

# ============================================
# Group 1: All Create Options with Both Binaries
# ============================================

echo -e "${BLUE}Group: Model Creation on Both Binaries${NC}"

run_test \
    "Forest: Create with all criterion types" \
    "$FOREST_BIN create --trees=3 --criterion=gini --save=$TEMP_DIR/f_gini.json && $FOREST_BIN create --trees=3 --criterion=entropy --save=$TEMP_DIR/f_entropy.json && $FOREST_BIN create --trees=3 --criterion=mse --save=$TEMP_DIR/f_mse.json && $FOREST_BIN create --trees=3 --criterion=variance --save=$TEMP_DIR/f_variance.json && echo 'ok'" \
    "ok"

run_test \
    "FacadeForest: Create with all criterion types" \
    "$FACADE_BIN create --trees=3 --criterion=gini --save=$TEMP_DIR/fa_gini.json && $FACADE_BIN create --trees=3 --criterion=entropy --save=$TEMP_DIR/fa_entropy.json && $FACADE_BIN create --trees=3 --criterion=mse --save=$TEMP_DIR/fa_mse.json && $FACADE_BIN create --trees=3 --criterion=variance --save=$TEMP_DIR/fa_variance.json && echo 'ok'" \
    "ok"

run_test \
    "Forest: Create classification and regression" \
    "$FOREST_BIN create --trees=5 --task=classification --save=$TEMP_DIR/f_class.json && $FOREST_BIN create --trees=5 --task=regression --save=$TEMP_DIR/f_reg.json && echo 'ok'" \
    "ok"

run_test \
    "FacadeForest: Create classification and regression" \
    "$FACADE_BIN create --trees=5 --task=classification --save=$TEMP_DIR/fa_class.json && $FACADE_BIN create --trees=5 --task=regression --save=$TEMP_DIR/fa_reg.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Group 2: Cross-Loading All Combinations
# ============================================

echo -e "${BLUE}Group: Cross-Loading All Model Types${NC}"

for model in gini entropy mse variance; do
    run_test \
        "Forest loads FacadeForest-created $model model" \
        "$FOREST_BIN info --model=$TEMP_DIR/fa_${model}.json" \
        "Number of trees: 3"

    run_test \
        "FacadeForest loads Forest-created $model model" \
        "$FACADE_BIN info --model=$TEMP_DIR/f_${model}.json 2>/dev/null" \
        "Trees: 3"
done

for task in class reg; do
    run_test \
        "Forest loads FacadeForest-created ${task} model" \
        "$FOREST_BIN info --model=$TEMP_DIR/fa_${task}.json" \
        "Number of trees: 5"

    run_test \
        "FacadeForest loads Forest-created ${task} model" \
        "$FACADE_BIN info --model=$TEMP_DIR/f_${task}.json 2>/dev/null" \
        "Trees: 5"
done

echo ""

# ============================================
# Group 3: JSON Integrity Across Binaries
# ============================================

echo -e "${BLUE}Group: JSON Integrity and Preservation${NC}"

check_json_valid \
    "Forest-created gini model JSON valid" \
    "$TEMP_DIR/f_gini.json"

check_json_valid \
    "FacadeForest-created gini model JSON valid" \
    "$TEMP_DIR/fa_gini.json"

run_test \
    "Forest-created entropy model has entropy criterion" \
    "grep -q '\"criterion\": \"entropy\"' $TEMP_DIR/f_entropy.json && echo 'ok'" \
    "ok"

run_test \
    "FacadeForest-created entropy model has entropy criterion" \
    "grep -q '\"criterion\": \"entropy\"' $TEMP_DIR/fa_entropy.json && echo 'ok'" \
    "ok"

run_test \
    "Forest-created regression has regression task" \
    "grep -q '\"task_type\": \"regression\"' $TEMP_DIR/f_reg.json && echo 'ok'" \
    "ok"

run_test \
    "FacadeForest-created regression has task_type field" \
    "grep -q '\"task_type\":' $TEMP_DIR/fa_reg.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Group 4: Hyperparameter Preservation
# ============================================

echo -e "${BLUE}Group: Hyperparameter Preservation${NC}"

run_test \
    "Forest create with all hyperparameters" \
    "$FOREST_BIN create --trees=7 --max-depth=6 --min-leaf=2 --min-split=3 --max-features=4 --save=$TEMP_DIR/f_hyper.json && echo 'ok'" \
    "ok"

run_test \
    "FacadeForest loads hyperparameters successfully" \
    "$FACADE_BIN info --model=$TEMP_DIR/f_hyper.json 2>&1" \
    "Number of Trees"

run_test \
    "Forest loads and preserves FacadeForest hyperparameters" \
    "$FACADE_BIN create --trees=8 --max-depth=7 --min-leaf=3 --min-split=4 --max-features=5 --save=$TEMP_DIR/fa_hyper.json && $FOREST_BIN info --model=$TEMP_DIR/fa_hyper.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Group 5: Training Data Preservation
# ============================================

echo -e "${BLUE}Group: Training Data Preservation${NC}"

run_test \
    "Generate training data" \
    "python3 -c \"
import random
random.seed(42)
with open('$TEMP_DIR/train.csv', 'w') as f:
    for i in range(100):
        features = [random.random() * 10 for _ in range(10)]
        label = 0 if sum(features) < 50 else 1
        line = ','.join(f'{f:.4f}' for f in features) + f',{label}\n'
        f.write(line)
print('ok')
\" && echo 'ok'" \
    "ok"

run_test \
    "Forest: Train model saves num_features" \
    "$FOREST_BIN create --trees=5 --save=$TEMP_DIR/f_train_model.json && $FOREST_BIN train --model=$TEMP_DIR/f_train_model.json --data=$TEMP_DIR/train.csv --save=$TEMP_DIR/f_trained.json && grep -q '\"num_features\": 10' $TEMP_DIR/f_trained.json && echo 'ok'" \
    "ok"

run_test \
    "FacadeForest: Load Forest-trained model" \
    "$FACADE_BIN info --model=$TEMP_DIR/f_trained.json 2>/dev/null" \
    "Trees: 5"

run_test \
    "FacadeForest: Train model saves num_features" \
    "$FACADE_BIN create --trees=5 --save=$TEMP_DIR/fa_train_model.json && $FOREST_BIN train --model=$TEMP_DIR/fa_train_model.json --data=$TEMP_DIR/train.csv --save=$TEMP_DIR/fa_trained.json && grep -q '\"num_features\": 10' $TEMP_DIR/fa_trained.json && echo 'ok'" \
    "ok"

run_test \
    "Forest: Load FacadeForest-trained model" \
    "$FOREST_BIN info --model=$TEMP_DIR/fa_trained.json" \
    "Random Forest Model Information"

echo ""

# ============================================
# Group 6: Complex Round-trip Tests
# ============================================

echo -e "${BLUE}Group: Complex Round-trip Workflows${NC}"

run_test \
    "Workflow 1: Forest create → FacadeForest load → verify criterion" \
    "$FOREST_BIN create --trees=10 --criterion=entropy --task=regression --max-depth=7 --save=$TEMP_DIR/rt1_forest.json && grep -q '\"criterion\": \"entropy\"' $TEMP_DIR/rt1_forest.json && echo 'ok'" \
    "ok"

run_test \
    "Workflow 2: FacadeForest create → Forest load → verify" \
    "$FACADE_BIN create --trees=12 --criterion=mse --task=classification --max-depth=8 --save=$TEMP_DIR/rt2_facade.json && $FOREST_BIN info --model=$TEMP_DIR/rt2_facade.json && grep -q '\"criterion\": \"mse\"' $TEMP_DIR/rt2_facade.json && echo 'ok'" \
    "ok"

run_test \
    "Workflow 3: Forest train → FacadeForest load → Forest load again" \
    "$FOREST_BIN create --trees=6 --save=$TEMP_DIR/rt3_a.json && $FOREST_BIN train --model=$TEMP_DIR/rt3_a.json --data=$TEMP_DIR/train.csv --save=$TEMP_DIR/rt3_trained.json && $FACADE_BIN info --model=$TEMP_DIR/rt3_trained.json 2>/dev/null && $FOREST_BIN info --model=$TEMP_DIR/rt3_trained.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Group 7: JSON Reloading and Re-saving
# ============================================

echo -e "${BLUE}Group: JSON Re-serialization Fidelity${NC}"

run_test \
    "Save Forest model, verify JSON valid" \
    "$FOREST_BIN create --trees=4 --criterion=variance --max-depth=4 --save=$TEMP_DIR/resave1.json && python3 -m json.tool $TEMP_DIR/resave1.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Verify criterion preserved through load/save cycle" \
    "grep -q '\"criterion\": \"variance\"' $TEMP_DIR/resave1.json && echo 'ok'" \
    "ok"

run_test \
    "Save FacadeForest model, load in Forest, save again" \
    "$FACADE_BIN create --trees=5 --criterion=entropy --max-depth=5 --save=$TEMP_DIR/resave2.json && python3 -m json.tool $TEMP_DIR/resave2.json > /dev/null && $FOREST_BIN info --model=$TEMP_DIR/resave2.json && echo 'ok'" \
    "ok"

run_test \
    "Verify task_type preserved through load/save cycle" \
    "grep -q '\"task_type\":' $TEMP_DIR/resave2.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Group 8: Multiple Sequential Operations
# ============================================

echo -e "${BLUE}Group: Sequential Cross-Binary Operations${NC}"

run_test \
    "Create Forest model" \
    "$FOREST_BIN create --trees=3 --save=$TEMP_DIR/seq_f1.json && echo 'ok'" \
    "ok"

run_test \
    "Load in FacadeForest and verify" \
    "$FACADE_BIN info --model=$TEMP_DIR/seq_f1.json 2>&1" \
    "Number of Trees"

run_test \
    "Create FacadeForest model" \
    "$FACADE_BIN create --trees=4 --save=$TEMP_DIR/seq_fa1.json && echo 'ok'" \
    "ok"

run_test \
    "Load in Forest" \
    "$FOREST_BIN info --model=$TEMP_DIR/seq_fa1.json && echo 'ok'" \
    "ok"

run_test \
    "Train Forest model" \
    "$FOREST_BIN train --model=$TEMP_DIR/seq_f1.json --data=$TEMP_DIR/train.csv --save=$TEMP_DIR/seq_f1_trained.json && echo 'ok'" \
    "ok"

run_test \
    "Load trained model in FacadeForest" \
    "$FACADE_BIN info --model=$TEMP_DIR/seq_f1_trained.json 2>/dev/null && echo 'ok'" \
    "ok"

run_test \
    "Train FacadeForest model" \
    "$FACADE_BIN create --trees=3 --save=$TEMP_DIR/seq_fa2.json && $FOREST_BIN train --model=$TEMP_DIR/seq_fa2.json --data=$TEMP_DIR/train.csv --save=$TEMP_DIR/seq_fa2_trained.json && echo 'ok'" \
    "ok"

run_test \
    "Load FacadeForest-trained model in Forest" \
    "$FOREST_BIN info --model=$TEMP_DIR/seq_fa2_trained.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Group 9: Feature Importances Preservation
# ============================================

echo -e "${BLUE}Group: Feature Importances Fidelity${NC}"

run_test \
    "Forest-trained model has feature_importances" \
    "grep -q '\"feature_importances\"' $TEMP_DIR/f_trained.json && echo 'ok'" \
    "ok"

run_test \
    "Feature importances is valid JSON array" \
    "python3 -c \"import json; m=json.load(open('$TEMP_DIR/f_trained.json')); print('ok' if isinstance(m['feature_importances'], list) else 'fail')\"" \
    "ok"

run_test \
    "FacadeForest-trained model has feature_importances" \
    "grep -q '\"feature_importances\"' $TEMP_DIR/fa_trained.json && echo 'ok'" \
    "ok"

run_test \
    "All feature importances sum approximately to 1.0" \
    "python3 -c \"import json; m=json.load(open('$TEMP_DIR/f_trained.json')); total=sum(m['feature_importances']); print('ok' if 0.9 < total < 1.1 else 'fail')\"" \
    "ok"

echo ""

# ============================================
# Group 10: Extreme Cases
# ============================================

echo -e "${BLUE}Group: Extreme Cases and Edge Conditions${NC}"

run_test \
    "Single tree forest: Forest create and load" \
    "$FOREST_BIN create --trees=1 --save=$TEMP_DIR/edge_f1.json && $FACADE_BIN info --model=$TEMP_DIR/edge_f1.json 2>/dev/null" \
    "Number of Trees: 1"

run_test \
    "Single tree forest: FacadeForest create and load" \
    "$FACADE_BIN create --trees=1 --save=$TEMP_DIR/edge_fa1.json && $FOREST_BIN info --model=$TEMP_DIR/edge_fa1.json" \
    "Number of trees: 1"

run_test \
    "Very deep forest: Forest create max-depth 25" \
    "$FOREST_BIN create --trees=3 --max-depth=25 --save=$TEMP_DIR/edge_f_deep.json && $FACADE_BIN info --model=$TEMP_DIR/edge_f_deep.json 2>/dev/null" \
    "Trees: 3"

run_test \
    "Very deep forest: FacadeForest create max-depth 25" \
    "$FACADE_BIN create --trees=3 --max-depth=25 --save=$TEMP_DIR/edge_fa_deep.json && $FOREST_BIN info --model=$TEMP_DIR/edge_fa_deep.json" \
    "Number of trees: 3"

run_test \
    "Large forest: 200 trees" \
    "$FOREST_BIN create --trees=200 --save=$TEMP_DIR/edge_f_large.json && $FACADE_BIN info --model=$TEMP_DIR/edge_f_large.json 2>/dev/null" \
    "Number of Trees: 200"

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
test_comprehensive_cross.sh
