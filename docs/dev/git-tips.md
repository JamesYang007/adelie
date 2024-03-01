# Git Tips

## Remove Large Files

This method comes from the following [StackOverflow page](https://stackoverflow.com/a/74592032).
First, install `git-filter-repo`:
```
pip install git-filter-repo
```

Next, follow these steps:
1. Run the following to display large files:
    ```bash
    git rev-list --objects --all | grep -f <(git verify-pack -v  .git/objects/pack/*.idx| sort -k 3 -n | cut -f 1 -d " " | tail -10)
    ```

2. Remove files following a regex pattern:
    ```bash
    git filter-repo --path-glob '<regex-pattern>' --invert-paths --force
    ```

3. Add remote to push to GitHub:
    ```bash
    git remote add origin git@github.com:JamesYang007/adelie.git
    ```

4. Finally, force push all the changes:
    ```bash
    git push --all --force
    git push --tags --force
    ```
