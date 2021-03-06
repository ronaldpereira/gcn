# python amlsim_process.py data/transactions_sample.csv data/ --filename amlsim_sample
# python train.py --dataset amlsim_sample --model gcn > output/amlsim_sample_gcn.txt
# python train.py --dataset amlsim_sample --model gcn_cheby > output/amlsim_sample_gcn_cheby.txt
# python train.py --dataset amlsim_sample --model dense > output/amlsim_sample_dense.txt

# python amlsim_process.py data/transactions.csv data/ --filename amlsim
# python train.py --dataset amlsim --model gcn > output/amlsim_gcn.txt
# python train.py --dataset amlsim --model gcn_cheby > output/amlsim_gcn_cheby.txt
# python train.py --dataset amlsim --model dense > output/amlsim_dense.txt

# python amlsim_process.py data/transactions_10k.csv data/ --filename amlsim_10k
# python train.py --dataset amlsim_10k --model gcn > output/amlsim_10k_gcn.txt
# python train.py --dataset amlsim_10k --model gcn_cheby > output/amlsim_10k_gcn_cheby.txt
# python train.py --dataset amlsim_10k --model dense > output/amlsim_10k_dense.txt

# python amlsim_process.py data/transactions_oversample.csv data/ --filename amlsim_oversample
# python train.py --dataset amlsim_oversample --model gcn > output/amlsim_oversample_gcn.txt
# python train.py --dataset amlsim_oversample --model gcn_cheby > output/amlsim_oversample_gcn_cheby.txt
# python train.py --dataset amlsim_oversample --model dense > output/amlsim_oversample_dense.txt

# python amlsim_process.py data/transactions_1k.csv data/ --filename amlsim_1k
# python train.py --dataset amlsim_1k --model gcn > output/amlsim_1k_gcn.txt
# python train.py --dataset amlsim_1k --model gcn_cheby > output/amlsim_1k_gcn_cheby.txt
# python train.py --dataset amlsim_1k --model dense > output/amlsim_1k_dense.txt

python amlsim_process.py data/transactions_1k_jan.csv data/ --filename amlsim_1k_jan
python train.py --dataset amlsim_1k_jan --model gcn > output/amlsim_1k_jan_gcn.txt
python train.py --dataset amlsim_1k_jan --model gcn_cheby > output/amlsim_1k_jan_gcn_cheby.txt
python train.py --dataset amlsim_1k_jan --model dense > output/amlsim_1k_jan_dense.txt
