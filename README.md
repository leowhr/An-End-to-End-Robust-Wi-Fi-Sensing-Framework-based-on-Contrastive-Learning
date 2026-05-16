### 运行Tensorboard

tensorboard --logdir=runs

python evaluate.py \
  --data-folder '/mnt/data/keran/project/WiSRL/dataset/10fenlei(5000)' \
  --model-module models.finetune_model_simCLR \
  --model-class WiSRL_tune \
  --model-kwargs '{"encoder_nlayers":3,"Proj_head":true,"proj_dim":64,"input_dim":270,"hidden_dim":64,"bimamba_type":"v2"}' \
  --ckpt ./model_weight/PRE/pre_100_Biblockv2_3+3link_test.pth