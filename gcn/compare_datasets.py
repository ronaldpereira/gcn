import pickle

with open('data/ind.amlsim_sample.tx', 'rb') as f:
    amlsim = pickle.load(f, encoding='latin1')

with open('data/ind.cora.tx', 'rb') as f:
    cora = pickle.load(f, encoding='latin1')

with open('data/ind.pubmed.tx', 'rb') as f:
    pubmed = pickle.load(f, encoding='latin1')

with open('data/ind.citeseer.tx', 'rb') as f:
    citeseer = pickle.load(f, encoding='latin1')

datasets = [amlsim, cora, pubmed, citeseer]

for ds in datasets:
    print(ds.shape)
    print(ds)
