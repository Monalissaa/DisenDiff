experiment_name='2023-11-23T11-18-37_full_1' # the trained experiment dir name
classes=cat_dog  # the input image
pretrained_model_path=checkpoints/v1-5-pruned.ckpt
datapath=datasets/images/$classes
newtoken=2 # the number of new tokens
seed=1 
save_path=DisenDiff/$classes


# save the changed params
python src/get_deltas.py --path $save_path/$experiment_name --newtoken $newtoken


prompts_global_path=datasets/prompts/${classes}_global.txt
prompts_subject1_path=datasets/prompts/${classes}_subject1.txt
prompts_subject2_path=datasets/prompts/${classes}_subject2.txt


global_file_name=val_global
python sample.py --delta_ckpt $save_path/$experiment_name/checkpoints/delta_epoch\=000004.ckpt --ckpt $pretrained_model_path --ddim_steps 50  --skip_grid --n_samples 4 --from-file $prompts_global_path --n_iter 4 --file_name $global_file_name

subject1_file_name=val_subject1
python sample.py --delta_ckpt $save_path/$experiment_name/checkpoints/delta_epoch\=000004.ckpt --ckpt $pretrained_model_path --ddim_steps 50  --skip_grid --n_samples 4 --from-file $prompts_subject1_path --n_iter 4 --file_name $subject1_file_name

subject2_file_name=val_subject2
python sample.py --delta_ckpt $save_path/$experiment_name/checkpoints/delta_epoch\=000004.ckpt --ckpt $pretrained_model_path --ddim_steps 50  --skip_grid --n_samples 4 --from-file $prompts_subject2_path --n_iter 4 --file_name $subject2_file_name





all_eval_file_path=$save_path/$experiment_name/global_subject1_subject2.txt
touch $all_eval_file_path
echo '#################################global###############################' >>  $all_eval_file_path

####  global
evaluate_path=$save_path/$experiment_name/evaluate_$global_file_name
sample_path=$save_path/$experiment_name/$global_file_name
mkdir $evaluate_path
real_data_dir=datasets/data_eval/${classes}/global

###################      text align score
file_count=0
while read line
do  
    ################################### change the logic
    # prompt=$(echo "$line" | sed 's/<new1>//g')
    prompt=$(echo "$line" | sed 's/<new[0-9]\+>//g' | tr -s ' ' | sed 's/^ //;s/ $//')
    filename_=${line// /_}  
    file_name=$(echo $filename_ | sed 's/ /_/g' | sed 's/<//g' | sed 's/>//g') 

    # file_name=$(echo $filename_ | sed 's/<background>//g') 

    mkdir $evaluate_path/$file_name
    dest_dir=$evaluate_path/$file_name
    file_count_max=$(echo "$file_count + 15" | bc)
    
    # for i in {0..49}
    for i in $(seq $file_count $file_count_max);
    do  
        image_name=$(printf "%05d" "$i").png
        cp "$sample_path/$image_name" "$dest_dir"
    done

    file_count_max_minus1=$(($file_count_max - 1))
    echo "{$(for i in $(seq $file_count $file_count_max_minus1);do image_name=$(printf "%05d" "$i"); echo "\"$image_name\":\"$prompt\",";done)" > $evaluate_path/$file_name.json
    image_name=$(printf "%05d" "$file_count_max"); echo "\"$image_name\":\"$prompt\"}" >> $evaluate_path/$file_name.json  # last json element
    
    python utils/clipscore/clipscore.py $evaluate_path/$file_name.json $evaluate_path/$file_name

    file_count=$(($file_count + 16))
done < $prompts_global_path


# read last line
line=$(tail -n 1 $prompts_global_path)
prompt=$(echo "$line" | sed 's/<new[0-9]\+>//g' | tr -s ' ' | sed 's/^ //;s/ $//')
# prompt=$(echo "$line" | sed 's/<new1>//g')
filename_=${line// /_}  
file_name=$(echo $filename_ | sed 's/ /_/g' | sed 's/<//g' | sed 's/>//g') 

mkdir $evaluate_path/$file_name
dest_dir=$evaluate_path/$file_name
file_count_max=$(echo "$file_count + 15" | bc)

# # for i in {0..49}
for i in $(seq $file_count $file_count_max);
do  
    image_name=$(printf "%05d" "$i").png
    cp "$sample_path/$image_name" "$dest_dir"
done

file_count_max_minus1=$(($file_count_max - 1))
echo "{$(for i in $(seq $file_count $file_count_max_minus1);do image_name=$(printf "%05d" "$i"); echo "\"$image_name\":\"$prompt\",";done)" > $evaluate_path/$file_name.json
image_name=$(printf "%05d" "$file_count_max"); echo "\"$image_name\":\"$prompt\"}" >> $evaluate_path/$file_name.json  # last json element

python utils/clipscore/clipscore.py $evaluate_path/$file_name.json $evaluate_path/$file_name


# !/bin/bash


content=$(cat $evaluate_path/text_alignment.txt)
arr=($content)

sum=0
for num in ${arr[@]} 
do
sum=$(echo "$sum + $num" | bc)
done

average=$(echo "$sum ${#arr[@]}" | awk '{print $1 / $2}')
echo $average >> $evaluate_path/text_alignment.txt


####################      image align score

python utils/clip_eval.py --real_data_dir $real_data_dir --fake_data_dir $evaluate_path --prompts_path $prompts_global_path




cat $evaluate_path/text_alignment.txt >> $all_eval_file_path







echo '#################################subject1###############################' >>  $all_eval_file_path
####  subject1
evaluate_path=$save_path/$experiment_name/evaluate_$subject1_file_name
sample_path=$save_path/$experiment_name/$subject1_file_name
mkdir $evaluate_path
real_data_dir=datasets/data_eval/${classes}/subject1

###################      text align score
file_count=0
while read line
do  
    ################################### change the logic
    # prompt=$(echo "$line" | sed 's/<new1>//g')
    prompt=$(echo "$line" | sed 's/<new[0-9]\+>//g' | tr -s ' ' | sed 's/^ //;s/ $//')
    filename_=${line// /_}  
    file_name=$(echo $filename_ | sed 's/ /_/g' | sed 's/<//g' | sed 's/>//g') 

    # file_name=$(echo $filename_ | sed 's/<background>//g') 

    mkdir $evaluate_path/$file_name
    dest_dir=$evaluate_path/$file_name
    file_count_max=$(echo "$file_count + 15" | bc)
    
    # for i in {0..49}
    for i in $(seq $file_count $file_count_max);
    do  
        image_name=$(printf "%05d" "$i").png
        cp "$sample_path/$image_name" "$dest_dir"
    done

    file_count_max_minus1=$(($file_count_max - 1))
    echo "{$(for i in $(seq $file_count $file_count_max_minus1);do image_name=$(printf "%05d" "$i"); echo "\"$image_name\":\"$prompt\",";done)" > $evaluate_path/$file_name.json
    image_name=$(printf "%05d" "$file_count_max"); echo "\"$image_name\":\"$prompt\"}" >> $evaluate_path/$file_name.json  # last json element
    
    python utils/clipscore/clipscore.py $evaluate_path/$file_name.json $evaluate_path/$file_name

    file_count=$(($file_count + 16))
done < $prompts_subject1_path


# read last line
line=$(tail -n 1 $prompts_subject1_path)
prompt=$(echo "$line" | sed 's/<new[0-9]\+>//g' | tr -s ' ' | sed 's/^ //;s/ $//')
# prompt=$(echo "$line" | sed 's/<new1>//g')
filename_=${line// /_}  
file_name=$(echo $filename_ | sed 's/ /_/g' | sed 's/<//g' | sed 's/>//g') 

mkdir $evaluate_path/$file_name
dest_dir=$evaluate_path/$file_name
file_count_max=$(echo "$file_count + 15" | bc)

# # for i in {0..49}
for i in $(seq $file_count $file_count_max);
do  
    image_name=$(printf "%05d" "$i").png
    cp "$sample_path/$image_name" "$dest_dir"
done

file_count_max_minus1=$(($file_count_max - 1))
echo "{$(for i in $(seq $file_count $file_count_max_minus1);do image_name=$(printf "%05d" "$i"); echo "\"$image_name\":\"$prompt\",";done)" > $evaluate_path/$file_name.json
image_name=$(printf "%05d" "$file_count_max"); echo "\"$image_name\":\"$prompt\"}" >> $evaluate_path/$file_name.json  # last json element

python utils/clipscore/clipscore.py $evaluate_path/$file_name.json $evaluate_path/$file_name


# !/bin/bash


content=$(cat $evaluate_path/text_alignment.txt)
arr=($content)

sum=0
for num in ${arr[@]} 
do
sum=$(echo "$sum + $num" | bc)
done

average=$(echo "$sum ${#arr[@]}" | awk '{print $1 / $2}')
echo $average >> $evaluate_path/text_alignment.txt


####################      image align score

python utils/clip_eval.py --real_data_dir $real_data_dir --fake_data_dir $evaluate_path --prompts_path $prompts_subject1_path




cat $evaluate_path/text_alignment.txt >> $all_eval_file_path







echo '#################################subject2###############################' >>  $all_eval_file_path




########################### subject2 

evaluate_path=$save_path/$experiment_name/evaluate_$subject2_file_name
sample_path=$save_path/$experiment_name/$subject2_file_name
mkdir $evaluate_path
real_data_dir=datasets/data_eval/${classes}/subject2

###################      text align score
file_count=0
while read line
do  
    ################################### change the logic
    # prompt=$(echo "$line" | sed 's/<new1>//g')
    prompt=$(echo "$line" | sed 's/<new[0-9]\+>//g' | tr -s ' ' | sed 's/^ //;s/ $//')
    filename_=${line// /_}  
    file_name=$(echo $filename_ | sed 's/ /_/g' | sed 's/<//g' | sed 's/>//g') 

    # file_name=$(echo $filename_ | sed 's/<background>//g') 

    mkdir $evaluate_path/$file_name
    dest_dir=$evaluate_path/$file_name
    file_count_max=$(echo "$file_count + 15" | bc)
    
    # for i in {0..49}
    for i in $(seq $file_count $file_count_max);
    do  
        image_name=$(printf "%05d" "$i").png
        cp "$sample_path/$image_name" "$dest_dir"
    done

    file_count_max_minus1=$(($file_count_max - 1))
    echo "{$(for i in $(seq $file_count $file_count_max_minus1);do image_name=$(printf "%05d" "$i"); echo "\"$image_name\":\"$prompt\",";done)" > $evaluate_path/$file_name.json
    image_name=$(printf "%05d" "$file_count_max"); echo "\"$image_name\":\"$prompt\"}" >> $evaluate_path/$file_name.json  # last json element
    
    python utils/clipscore/clipscore.py $evaluate_path/$file_name.json $evaluate_path/$file_name

    file_count=$(($file_count + 16))
done < $prompts_subject2_path


# read last line
line=$(tail -n 1 $prompts_subject2_path)
prompt=$(echo "$line" | sed 's/<new[0-9]\+>//g' | tr -s ' ' | sed 's/^ //;s/ $//')
# prompt=$(echo "$line" | sed 's/<new1>//g')
filename_=${line// /_}  
file_name=$(echo $filename_ | sed 's/ /_/g' | sed 's/<//g' | sed 's/>//g') 

mkdir $evaluate_path/$file_name
dest_dir=$evaluate_path/$file_name
file_count_max=$(echo "$file_count + 15" | bc)

# # for i in {0..49}
for i in $(seq $file_count $file_count_max);
do  
    image_name=$(printf "%05d" "$i").png
    cp "$sample_path/$image_name" "$dest_dir"
done

file_count_max_minus1=$(($file_count_max - 1))
echo "{$(for i in $(seq $file_count $file_count_max_minus1);do image_name=$(printf "%05d" "$i"); echo "\"$image_name\":\"$prompt\",";done)" > $evaluate_path/$file_name.json
image_name=$(printf "%05d" "$file_count_max"); echo "\"$image_name\":\"$prompt\"}" >> $evaluate_path/$file_name.json  # last json element

python utils/clipscore/clipscore.py $evaluate_path/$file_name.json $evaluate_path/$file_name


# !/bin/bash


content=$(cat $evaluate_path/text_alignment.txt)
arr=($content)

sum=0
for num in ${arr[@]} 
do
sum=$(echo "$sum + $num" | bc)
done

average=$(echo "$sum ${#arr[@]}" | awk '{print $1 / $2}')
echo $average >> $evaluate_path/text_alignment.txt


####################      image align score

python utils/clip_eval.py --real_data_dir $real_data_dir --fake_data_dir $evaluate_path --prompts_path $prompts_subject2_path


cat $evaluate_path/text_alignment.txt >> $all_eval_file_path






############### global 
evaluate_path=$save_path/$experiment_name/evaluate_$global_file_name
result_file_path=$evaluate_path/text_alignment.txt

file=$result_file_path

# 提取倒数第三行的数值
global_num=$(tail -n 3 "$file" | head -n 1)

# 提取倒数第二行的字符串
str=$(tail -n 2 "$file" | head -n 1)

# 提取字符串的第二个子串并转换为数值
second_substring=$(echo "$str" | cut -d ' ' -f 3)
global_num2=$(echo "$second_substring" | tr -d '[:alpha:]')


############### subject1 
evaluate_path=$save_path/$experiment_name/evaluate_$subject1_file_name
result_file_path=$evaluate_path/text_alignment.txt

file=$result_file_path

# 提取倒数第三行的数值
subject1_num=$(tail -n 3 "$file" | head -n 1)

# 提取倒数第二行的字符串
str=$(tail -n 2 "$file" | head -n 1)

# 提取字符串的第二个子串并转换为数值
second_substring=$(echo "$str" | cut -d ' ' -f 3)
subject1_num2=$(echo "$second_substring" | tr -d '[:alpha:]')




############### subject2
evaluate_path=$save_path/$experiment_name/evaluate_$subject2_file_name
result_file_path=$evaluate_path/text_alignment.txt

file=$result_file_path

# 提取倒数第三行的数值
subject2_num=$(tail -n 3 "$file" | head -n 1)

# 提取倒数第二行的字符串
str=$(tail -n 2 "$file" | head -n 1)

# 提取字符串的第二个子串并转换为数值
second_substring=$(echo "$str" | cut -d ' ' -f 3)
subject2_num2=$(echo "$second_substring" | tr -d '[:alpha:]')


echo -e "Image-alignment for combined, concept1, concept2:$global_num2\t$subject1_num2\t$subject2_num2" >> "$all_eval_file_path"
echo -e "Text-alignment for combined, concept1, concept2:$global_num\t$subject1_num\t$subject2_num" >> "$all_eval_file_path"

echo -e "$global_num2\t$subject1_num2\t$subject2_num2\t$global_num\t$subject1_num\t$subject2_num" >> "$all_eval_file_path"