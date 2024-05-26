# install homebrew + libomp
if [[ "$OSTYPE" == "darwin"* ]]; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install libomp

    # See https://cibuildwheel.pypa.io/en/stable/faq/#macos-library-dependencies-do-not-satisfy-target-macos
    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        echo "yay"
        export MACOSX_DEPLOYMENT_TARGET=14.0
    else
        echo "no yay"
        export MACOSX_DEPLOYMENT_TARGET=13
    fi
fi