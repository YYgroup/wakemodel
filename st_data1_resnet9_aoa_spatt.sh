python -u train.py -net resnet9_aoa_spatt -train_batch 32 \
      -dataset stationary_plate \
      -data_path ./data/stationary_plate/npy_files \
      -warm 1 \
      -lr 0.002 \
      -save_pre \
      -aoa \
      -train_val_type 1\
      -exp_name run2_st_data1 \
      -img_size  256
