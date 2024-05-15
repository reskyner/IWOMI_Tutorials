FROM tensorflow/tensorflow:2.15.0
RUN apt-get update
RUN apt-get install -y git-all
RUN git clone https://github.com/Kohulan/IWOMI_Tutorials
WORKDIR IWOMI_Tutorials
RUN python3 -m pip install -U pip
RUN pip install -r requirements.txt

