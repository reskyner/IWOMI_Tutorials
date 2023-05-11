# DECIMER IWOMI Tutorial
DECIMER Extraction Handon tutotial @ IWOMI 2023

### Instructions

### We suggest using DECIMER inside a Conda environment, which makes the dependencies install easily.
- Conda can be downloaded as part of the [Anaconda](https://www.anaconda.com/) or the [Miniconda](https://conda.io/en/latest/miniconda.html) platforms (Python 3.7). We recommend installing miniconda3. Using Linux, you can get it with:
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

#### Use a conda environment for clean installation
```shell
git clone https://github.com/Kohulan/DECIMER_IWOMI_Tutorial.git
cd DECIMER_IWOMI_Tutorial
conda create --name DECIMER python=3.10.0
conda activate DECIMER
conda install pip
python3 -m pip install -U pip
pip install -r requirements.txt
```

#### For Mac M1 users install JDK using this
```shell
https://www.azul.com/downloads/?os=macos&architecture=arm-64-bit&package=jdk#zulu
```
