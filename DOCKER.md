# Running tutorial in docker
## Intro
If you have any problems with installing locally with conda, you can also setup the tutorial material inside of a docker container, and either mount your local filesystem or copy files out of the container

First either clone this repository, or make a local copy of just the dockerfile (the code is cloned into the container image for installation)

## Using the container

### Running the container
First, to build the container:
```docker build -t <name_for_image> ```

Then find the image name or tag with `docker images` which will list any images you have built. e.g.:

```
docker images
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
<none>       <none>    4a50c3b8607d   3 hours ago   5.47GB
```

Then you can use the `TAG` if you have set one, or the `IMAGE ID` to enter the docker container. Because we are running a notebook, we also need to map port 8888 to our local port.

To enter the container run:
```docker exec -p 8888:8888 -it <tag/image_id> /bin/bash```

### Running the jupyter notebook server from inside the container

You should then be inside of the container in the IWOMI_Tutorial directory. You can run a jupyter notebook from here with:

```jupyter notebook --ip 0.0.0.0 --no-browser --allow-root```

Or with the server command instead if you prefer. 

You can now access the notebook in your preferred browser at `localhost:8888`. You can copy the token/url directly from the container and just replace the relevant part with `localhost` if you want. 

### Running the notebooks

Do this in the normal way as if you were running locally. 

### Retreiving files from the container 

Unless you have mounted your local filesystem into the container (overwriting the containers copy of this repo), then once you have run the tutorial notebooks, you will need to get them out of the container. We can do this with `docker cp`. All commands in the folder (locally) you want to copy files into:

```docker cp <tag/image_id>:/IWOMI_Tutorials/STOUT_Training/tokenizer_iupac.pkl .
docker cp <tag/image_id>:/IWOMI_Tutorials/STOUT_Training/tokenizer_cmiles.pkl .
docker cp <tag/image_id>:/IWOMI_Tutorials/STOUT_Training/Traning_data .```
