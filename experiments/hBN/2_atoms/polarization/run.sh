
python -u ../../../../trainer.py \
    --filename hbn_2_atom_data.h5 \
    --epochs 10000 \
    --rcut 3.5 \
    --split 0.8 \
    --batch_size 8 \
    --n_outputs 3 \
    --max_neighbors 500 \
    --image_shape 15 15 15  \
    --sigma 0.5 \
    --learning_rate 0.00001
