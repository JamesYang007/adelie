# install homebrew + libomp
if [[ "$OSTYPE" == "darwin"* ]]; then
    # See https://github.com/scikit-learn/scikit-learn/blob/09781c540077a7f1f4f2392c9287e08e479c4f29/build_tools/wheels/build_wheels.sh#L18-L50
    # See https://cibuildwheel.pypa.io/en/stable/faq/#macos-library-dependencies-do-not-satisfy-target-macos
    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        # install conda
        set -ex
        # macos arm64 runners do not have conda installed. Thus we much install conda manually
        EXPECTED_SHA="dd832d8a65a861b5592b2cf1d55f26031f7c1491b30321754443931e7b1e6832"
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Mambaforge-23.11.0-0-MacOSX-arm64.sh"
        curl -L --retry 10 $MINIFORGE_URL -o miniforge.sh

        # Check SHA
        file_sha=$(shasum -a 256 miniforge.sh | awk '{print $1}')
        if [ "$EXPECTED_SHA" != "$file_sha" ]; then
            echo "SHA values did not match!"
            exit 1
        fi

        # Install miniforge
        MINIFORGE_PATH=$HOME/miniforge
        bash ./miniforge.sh -b -p $MINIFORGE_PATH
        echo "$MINIFORGE_PATH/bin" >> $GITHUB_PATH
        echo "CONDA_HOME=$MINIFORGE_PATH" >> $GITHUB_ENV
         
        PATH="$PATH:$MINIFORGE_PATH/bin"

        export MACOSX_DEPLOYMENT_TARGET=12.0
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
    else
        # Non-macos arm64 envrionments already have conda installed
        echo "CONDA_HOME=/usr/local/miniconda" >> $GITHUB_ENV

        export MACOSX_DEPLOYMENT_TARGET=10.9
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
    fi

    sudo conda create -n build $OPENMP_URL
    OPENMP_PREFIX="$CONDA_HOME/envs/build"
fi