df

CONFIG=$1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh tools/dist_train.sh configs/$CONFIG.py 8