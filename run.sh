classes=cat_dog  # the input image
pretrained_model_path=checkpoints/v1-5-pruned.ckpt
datapath=datasets/images/$classes
caption="<new1> cat and <new2> dog"  # the training input prompt
modifier_token="<new1>+<new2>" # new tokens
newtoken=2 # the number of new tokens
fine_yaml=full.yaml
seed=1 
suffix="full_${seed}"
name=$classes_${suffix}
save_path=DisenDiff/$classes
class_prompt="a cat and a dog"


python src/retrieve.py --class_prompt "$class_prompt" --class_data_dir real_reg/samples_${classes}

python -u  train.py \
        --base configs/$fine_yaml  \
        -t --gpus=1 \
        --resume-from-checkpoint-custom  $pretrained_model_path \
        --caption "$caption" \
        --datapath $datapath \
        --reg_datapath "real_reg/samples_${classes}/images.txt" \
        --reg_caption "real_reg/samples_${classes}/caption.txt" \
        --modifier_token "$modifier_token" \
        --name "$name" \
        --logdir $save_path \
        --accumulate_grad_batches 4 \
        --seed $seed
