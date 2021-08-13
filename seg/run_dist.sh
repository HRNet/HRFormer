CONFIG=$1
MODE=$2
TAG=$3
python -m pip install timm
python -m pip install einops
pip uninstall PIL
pip install pillow
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/$CONFIG.sh $MODE $TAG .
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/$CONFIG.sh resume run0_ .