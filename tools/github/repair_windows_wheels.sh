#!/bin/bash
# See https://github.com/scikit-learn/scikit-learn/blob/main/build_tools/github/repair_windows_wheels.sh
set -e
set -x

WHEEL=$1
DEST_DIR=$2

# By default, the Windows wheels are not repaired.
# In this case, we need to vendor VCRUNTIME140.dll
pip install wheel
wheel unpack "$WHEEL"
WHEEL_DIRNAME=$(ls -d adelie-*)
python tools/github/vendor.py "$WHEEL_DIRNAME"
wheel pack "$WHEEL_DIRNAME" -d "$DEST_DIR"
rm -rf "$WHEEL_DIRNAME"