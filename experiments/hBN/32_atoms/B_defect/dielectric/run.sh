
python ../../../../trainer.py \
    --filename hBN_B_defect.h5 \
    --epochs 10000 \
    --rcut 3.5 \
    --split 0.8 \
    --batch_size 8 \
    --n_outputs 6 \
    --max_neighbors 500 \
    --image_shape 27 27 27  \
    --sigma 0.5 \
    --learning_rate 0.00001
