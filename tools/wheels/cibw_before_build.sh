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
elif [[ "$OSTYPE" == "linux"* ]]; then
    # Linux build is fully self-contained, so we must setup everything from scratch.
    # Install wget
    apt update
    apt upgrade
    apt install wget

    # Install miniconda
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    export PATH=$PATH:~/miniconda3/bin
    conda init bash

    # Create adelie environment
    conda create -n adelie eigen==3.4.0
    conda activate adelie
fi