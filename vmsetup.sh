export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt update
sudo apt-get install gcsfuse
sudo apt install vim

sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt install python3.7-dev
sudo apt install python3-pip
sudo -H python3.7 -m pip install --upgrade pip

gcloud init

#install requirements
sudo -H python3.7 -m pip install -r requirements.txt
sudo apt-get install libhdf5-serial-dev