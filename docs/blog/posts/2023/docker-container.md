---
tags: 
    - Development
    - Data Analysis
comments: true
---
# Working within Docker container

This has been my workflow to iteractively work from within a running container. 

The goal here is to *still* rely on the local machine for editing files but to rely on the environment from the container for running any code. 

This simple setup provides: 

- Setup of environment regardless of current system dependencies
- Flexible and reproducible environments
- Access to local files from container's shell 

In short, the process is as follows: 

1. Find or build base image
2. Open container's shell 

```bash 
IMAGE_NAME=quick-env 
docker build -t $IMAGE_NAME . 

docker run --rm -it -v $(pwd):/app -w /app --entrypoint bash $IMAGE_NAME
```

Below is a breakdown of these commands. 

## Find or build base image

We can use one of the many images on [Docker Hub](https://hub.docker.com/search?q=) or build off one with a custom `Dockerfile`. 

Below has python 3.11 with `pandas`, `matplotlib`, and `IPython`  installed: 

```Dockerfile title="Dockerfile"
FROM python:3.11

# Setup the environment to use in terminal
RUN pip install pandas matplotlib IPython
```

Build a new image with `docker build -t <image-name> .`

!!! note 
    Nothing from the local file system is tranferred here as we will still rely on the local file system

## Run container's shell

Run this container and enter iteractively into its shell with the following command: 

`docker run --rm -it -v $(pwd):/app -w /app --entrypoint bash <image-name>`

These flag shouldn't need to be changed and should work in most cases but can always be customized. 

The [`docker run` documentation](https://docs.docker.com/engine/reference/commandline/run/) is very thorough but here is some info on the flags used: 

### `--rm`: Container cleanup

This is optional flag but is helpful for decluttering after a run.

### `-it`: Interative terminal

Use the `-i` and `-t` flags (or `-it` together) in order for running container to use our input and output. 
I alway remember this as **i**teractive **t**erminal

### `-v`: Access to local files

Mount local volume such that changes to local files can be accessed from and by the container. 

This can be any location on either local or image, but `$(pwd):/app` usually does the trick.

!!! note
    The `/app` location will be created if it doesn't already exist!

### `-w`: Setting container working dir

Though this is not required, this will set the local directory for the container to where we mounted our local files. 

### `--entrypoint`: Enter container terminal

Using `docker run` doesn't guarantee that a terminal will be kicked off. However, the entrypoint for the image can be overridden during the with the `--entrypoint` flag. 

!!! tip 
    Most of the time the shell will be `bash` but could be another shell like `sh`

## Summary

Docker makes it easy to create new isolated environments without touching local system dependencies while working with local files.

Overall, Docker is awesome :smile:

## References 

- [Docker Hub](https://hub.docker.com/search?q=)
- [`docker run` Documentation](https://docs.docker.com/engine/reference/commandline/run/)
