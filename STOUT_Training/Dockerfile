FROM tensorflow/tensorflow:2.15.0
RUN apt-get update
RUN apt-get install -y git-all
RUN git clone https://github.com/Kohulan/IWOMI_Tutorials
WORKDIR IWOMI_Tutorials
RUN python3 -m pip install -U pip
RUN pip install -r requirements.txt

# # OPTIONAL: install google-cloud-sdk
# # Add the Cloud SDK distribution URI as a package source
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# # Import the Google Cloud Platform public key
# RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# # Update the package list
# RUN apt-get update

# # Install the Google Cloud SDK
# RUN apt-get install -y google-cloud-sdk
