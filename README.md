# ESR-MM
Official pytorch implementation of Leveraging Enriched Skeleton Representation with Multi-relational Metrics for Few-shot Action Recognition
# Environment
pytorch = 1.13.1

python = 3.9.16

tqdm = 4.65.0

pynvml = 11.5.0

fitlog = 0.9.15
# Datasets
NTU-T, NTU-S, and Kinetics are composed of [https://github.com/NingMa-AI/DASTM](https://github.com/NingMa-AI/DASTM) Provide.

NTU RGB+D 120 one-shot setting dataset: https://pan.baidu.com/s/1bvJhOeHRGyKWfuSKNJZGJQ?pwd=dpes

https://drive.google.com/file/d/1qaMhvwBqTfuLViv3M9neuLTnqoz6RDHF/view?usp=sharing

# Run
Run our proposed method on the NTU-T dataset with 5-way 1-shot settings:
```python
python train.py   -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003
```
Run our proposed method on the NTU-T dataset with 5-way 5-shot settings:
```python
python train.py   -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003 -nsTr 5 -nsVa 5
```
Pass in different parameters to run different datasets:
```python
--dataset    ntu-T
--dataset    ntu-S
--dataset    kinetics
--dataset    ntu1s
```
Run our proposed method on the NTU RGB+D 120 one-shot setting dataset:
```python
python train.py    --dataset ntu1s   -cTr 20 -nqTr 5 -cVa 20 -nqVa 5  --epochs 300 -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003
```
Using different backbones:
```python
python train.py  --bakbone ctrgcn
python train.py  --bakbone stgcnpp
python train.py  --bakbone hdgcn
```
