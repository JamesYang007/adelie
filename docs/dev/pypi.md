# PyPI Notes

## Setting up Credentials

Uploading to `PyPI` requires some credentials.
Create the file `$HOME/.pypirc` containing the following:
```
[pypi]
  username = __token__
  password = ...
```
where the `...` is replaced with the API token given [here](https://github.com/JamesYang007/JamesYang007.github.io/blob/main/secrets/pypi_api_token.txt).

## Versioning

Specify the version in [VERSION](../../VERSION).
Make sure to update this file before a release!

## Git Tag

Make sure to tag the commit before a release!
```
git tag -a vx.x.x -m "vx.x.x"
git push --tags
```

## Uploading to PyPI

We assume the reader is in the root directory of the repository.

1. Run the following to package the `tar` file:
    ```
    python setup.py sdist
    ```
    This creates a folder called `dist` in the root directory
    containing the `tar.gz` file.

2. Upload to PyPI:
    ```
    twine upload dist/*x.x.x.tar.gz
    ```
    where `x.x.x` is the version number.