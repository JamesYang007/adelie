# Development Workflow

## Package Build

1. Install the package in editable mode:
    ```bash
    pip install -e . --verbose
    ```

2. For subsequent builds for the C++ backend, run the following:
    ```bash
    CC="ccache <c-compiler>" CXX="ccache <cpp-compiler>" python setup.py build_ext --inplace --force
    ```
    For example, on MacOS using `clang`, the command would be
    ```bash
    CC="ccache clang" CXX="ccache clang++" python setup.py build_ext --inplace --force
    ```

3. For testing the release build, run step 1.
    The flags used for the linker are different between the two steps.
    The former ensures that OpenMP is linked properly.