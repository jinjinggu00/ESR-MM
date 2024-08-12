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

# Data Storage Format

Our data is organized in the following folder structure:

- data/
  - kinetics/
    - Kinetics/
  - ntu1s/
  - ntu120/
    - NTU-S/
    - NTU-T/

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
python train.py  --backbone ctrgcn
python train.py  --backbone stgcnpp
python train.py  --backbone hdgcn
```

# Train using multimodal approaches (bond, joint speed, and bond speed)
Train using the Early Fusion 1 method mentioned in the paper:
```python
python train.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003 --modal 2
```
Train using the Early Fusion 2 method mentioned in the paper:
```python
python train.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003 --modal 4 --weighted 1 
```
Train using the Early Fusion 3 method mentioned in the paper:
```python
python train_2ab.py -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003 --modal 4 --process 1
```
Please note the "--modal" parameter in the script, which can be set to 2-4 to obtain results using 2-4 modalities. The other parameters do not need to be changed. If results for a 5-way 5-shot setting are required, please add "-nsTr 5 -nsVa 5" to the script.

In the paper, it is mentioned that the late fusion method requires training multiple models. We recommend using a shell script to complete the training process. Create a script file named "run.sh" with the following content:
```shell
#!/bin/bash
python train2.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003  -model_name esr 
wait

python train2.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003  -model_name esr -bone 1 
wait

python train2.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003  -model_name esr -vel 1 
wait

python train2.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003  -model_name esr -bone 1 -vel 1 
wait

python ensemble.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003  -model_name esr --modal 2
wait

python ensemble.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003  -model_name esr --modal 3
wait

python ensemble.py  -reg 0.0005 --AA 1 --metric tcmhm --pca 0.003  -model_name esr --modal 4
wait
```
The late fusion method involves first completing the training of four individual models, and then using the "ensemble.py" file to obtain the final accuracy using different modalities, which can be achieved by adjusting the "--modal" parameter. If results for a 5-way 5-shot setting are needed, add "-nsTr 5 -nsVa 5" to each line in the script.

## Comprehensive documentation and detailed usage instructions are in progress. The codebase is currently undergoing refinement and optimization. We appreciate your patience as we work to provide a more robust and well-documented implementation. Updates will be made available as soon as possible.

# Acknowledgements
This repository is based on [DASTM](https://github.com/NingMa-AI/DASTM).

We appreciate the original authors for their significant contributions.
