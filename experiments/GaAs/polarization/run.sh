
python ../../../trainer.py \
    --filename GaAs_data.h5 \
    --epochs 10000 \
    --rcut 3.5 \
    --split 0.8 \
    --batch_size 8 \
    --n_outputs 3 \
    --max_neighbors 500 \
    --image_shape 27 27 27  \
    --sigma 1.0 \
    --learning_rate 0.00001
