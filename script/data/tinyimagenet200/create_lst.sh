#! /bin/zsh
mxd=/home/peiyunh/mxnet
cwd=$(pwd)
cd $cwd/data/tinyimagenet200
python $mxd/tools/im2rec.py --list True --shuffle True --recursive True --num-thread 8 train train
awk ' { t = $3; $3 = int($2); $2 = "./train/"t; print; } ' train.lst >> train.lst
# NOTE: we do not need to create record here
# python $mxd/tools/im2rec.py --num-thread 16 train.lst train 
python $mxd/tools/im2rec.py --list True --shuffle True --recursive True --num-thread 8 val val
awk ' { t = $3; $3 = int($2); $2 = "./val/"t; print; } ' val.lst >> val.lst
# NOTE: we do not need to create record here
# python $mxd/tools/im2rec.py --num-thread 16 val.lst val 
cd $cwd
