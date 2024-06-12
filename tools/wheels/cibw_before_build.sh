set -ex

if [[ "$OSTYPE" == "darwin"* ]]; then
    # See https://github.com/scikit-learn/scikit-learn/blob/09781c540077a7f1f4f2392c9287e08e479c4f29/build_tools/wheels/build_wheels.sh#L18-L50
    # See https://cibuildwheel.pypa.io/en/stable/faq/#macos-library-dependencies-do-not-satisfy-target-macos
    # We need to explicitly set the deployment target to the version of the intended machine.
    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        export MACOSX_DEPLOYMENT_TARGET=12.0
    else
        export MACOSX_DEPLOYMENT_TARGET=10.9
    fi
fi

echo $MAMBA_ROOT_PREFIX
ls $MAMBA_ROOT_PREFIX/envs/adelie