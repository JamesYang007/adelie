# Building Wheels

This is a short primer on how to build wheels properly.
We will use the `cibuildwheels` tool, which seamlessly allows for building wheels
for different machines and Python versions.
Moreover, it is a docker-based method, so it is completely reproducible.

## Local Development

It is often convenient to use `cibuildwheels` locally and never interact with CI tools until the end.
Following the instructions on the [cibuildwheels documentation](https://cibuildwheel.readthedocs.io/en/stable/setup/),
run the following to first install cibuildwheel:

```bash
pip install cibuildwheel
```

Next, install `docker` from the [Docker installation page](https://docs.docker.com/engine/install/).
After installing `docker`, start the server:

```bash
sudo dockerd
```

Finally, to allow users with non-root privilege to use the server,
run the following:

```bash
# create docker group
sudo groupadd docker
# add user to docker group
sudo usermod -aG docker ${USER}
# re-log for effect to take place
su -s ${USER}
# verify docker can be run
docker run hello-world
# set permission for non-root users
sudo chmod 666 /var/run/docker.sock
```

After this, you can run `cibuildwheel` in the workspace directory.
For example, on a linux host machine, the following
will run `cibuildwheel` to build the wheel for `cp39-manylinux_x86_64`.

```bash
CIBW_BUILD=cp39-manylinux_x86_64 cibuildwheel --platform linux 
```