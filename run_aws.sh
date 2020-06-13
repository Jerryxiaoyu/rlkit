#!/usr/bin/env bash

PRO_DIR=/home/ubuntu/jerry
RLKIT_PATH=/home/ubuntu/jerry/rlkit
MULTIWORLD_PATH=/home/ubuntu/jerry/multiworld

source activate pytorch_p36

cd /home/ubuntu
sh install.sh 410.104
nvidia-smi



cd $PRO_DIR
#git clone -b my_master https://github.com/Jerryxiaoyu/rlkit.git
git clone -b my_branch https://github.com/Jerryxiaoyu/multiworld.git

cd $MULTIWORLD_PATH
git pull
pip install -r requirements.txt
python setep.py develop

cd $RLKIT_PATH
git pull


## run your code
