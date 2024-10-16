# Development Workflow

## Package Build

1. __(Single run)__ Install the package in editable mode the first time:
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