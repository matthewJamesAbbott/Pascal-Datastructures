#!/bin/bash
#
# MIT License
# Copyright (c) 2025 Matthew Abbott
#
# gan_facade_tests.sh - Bash test runner for GANFacade --test <function-name>
# Tests every GF_ function individually and reports results.
#
# Usage:
#   ./gan_facade_tests.sh              Run all tests
#   ./gan_facade_tests.sh --quick      Skip slow tests (GF_Train_Full, GF_Sec_RunFuzzTests)
#   ./gan_facade_tests.sh --category ops     Run only GF_Op_ tests
#   ./gan_facade_tests.sh --category gen     Run only GF_Gen_ tests
#   ./gan_facade_tests.sh --category disc    Run only GF_Disc_ tests
#   ./gan_facade_tests.sh --category train   Run only GF_Train_ tests
#   ./gan_facade_tests.sh --category sec        Run only GF_Sec_ tests
#   ./gan_facade_tests.sh --category introspect Run only GF_Introspect_ tests
#

set -euo pipefail

BINARY="./GANFacade"
PASS=0
FAIL=0
SKIP=0
TOTAL=0
QUICK=0
CATEGORY="all"
FAILURES=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --category) CATEGORY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--category ops|gen|disc|train|sec|introspect]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Slow tests to skip in quick mode
SLOW_TESTS="GF_Train_Full GF_Sec_RunFuzzTests GF_Sec_RunTests"

# Check binary exists
if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY not found or not executable."
    echo "Compile first: make GANFacade"
    exit 1
fi

# Get function list from binary
FUNCTIONS=$($BINARY --list)

run_test() {
    local func="$1"

    # Category filter
    case "$CATEGORY" in
        ops)        [[ "$func" != GF_Op_* ]] && return ;;
        gen)        [[ "$func" != GF_Gen_* ]] && return ;;
        disc)       [[ "$func" != GF_Disc_* ]] && return ;;
        train)      [[ "$func" != GF_Train_* ]] && return ;;
        sec)        [[ "$func" != GF_Sec_* ]] && return ;;
        introspect) [[ "$func" != GF_Introspect_* ]] && return ;;
        all)        ;;
    esac

    # Quick mode skip
    if [ "$QUICK" -eq 1 ]; then
        for slow in $SLOW_TESTS; do
            if [ "$func" = "$slow" ]; then
                printf "  [SKIP] %s (--quick)\n" "$func"
                ((SKIP++)) || true
                return
            fi
        done
    fi

    ((TOTAL++)) || true

    # Run the test, capture output and exit code
    local output
    output=$($BINARY --test "$func" 2>&1) && local rc=0 || local rc=$?

    if [ "$rc" -eq 0 ]; then
        printf "  [PASS] %s\n" "$func"
        ((PASS++)) || true
    else
        printf "  [FAIL] %s\n" "$func"
        ((FAIL++)) || true
        FAILURES="$FAILURES  - $func\n"
    fi
}

echo "================================================================="
echo " GANFacade Test Runner"
echo " Binary: $BINARY"
echo " Category: $CATEGORY"
echo " Quick mode: $([ "$QUICK" -eq 1 ] && echo "yes" || echo "no")"
echo "================================================================="
echo ""

# Run each test
while IFS= read -r func; do
    run_test "$func"
done <<< "$FUNCTIONS"

# Summary
echo ""
echo "================================================================="
echo " RESULTS: $TOTAL tested | $PASS passed | $FAIL failed | $SKIP skipped"
echo "================================================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    printf "$FAILURES"
    echo ""
    exit 1
else
    echo "All tests passed."
    exit 0
fi
