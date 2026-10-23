#!/bin/bash
# Usage: ./run.sh script.py [args]
# xgboost/lightgbm wheels need libomp.dylib, which is not installed system-wide;
# point dyld at the copy bundled inside the scikit-learn wheel of the same env.
SP=/private/tmp/claude-501/-Users-aretelew-Developer-csds-340/39cafc5a-5284-4393-84f2-4fab8d74f2fb/scratchpad
cd "$SP/trees" || exit 1
UVRUN="uv run -q --no-project --with numpy --with scipy --with scikit-learn --with xgboost --with lightgbm --with catboost"
OMP=$($UVRUN python -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))")
export DYLD_FALLBACK_LIBRARY_PATH="$OMP"
export PYTHONPATH="$SP:$SP/trees"
export PYTHONUNBUFFERED=1
exec $UVRUN python "$@"
