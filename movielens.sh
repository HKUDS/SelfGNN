CUDA_VISIBLE_DEVICES=0 python main.py --data movielens --lr 1e-3 --reg 1e-2 --ssl_reg 1e-6 --save_path movie6 --epoch 150  --batch 512 --sampNum 40 --sslNum 90 --graphNum 6 --gnn_layer 2 --att_layer 3 --test True  --testSize 1000 --ssldim 48 --keepRate 0.5 --pos_length 200 --leaky 0.5