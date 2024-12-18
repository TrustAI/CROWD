# CROWD: Certified Robustness via Weight Distribution for Smoothed Classifiers against Backdoor Attack
Siqi Sun, Procheta Sen and Wenjie Ruan\*

*Corresponding Author

A certified robustness verification is proposed with theoretical guarantees targeted to the NLP classifier against backdoor attacks.

##  Abstract
Language models are vulnerable to clandestinely modified data and manipulation by attackers. Despite considerable research dedicated to enhancing robustness against adversarial attacks, the realm of provable robustness for backdoor attacks remains relatively unexplored. In this paper, we initiate a pioneering investigation into the certified robustness of NLP models against backdoor triggers. We propose a model-agnostic mechanism for largescale models that applies to complex model structures without the need to assess model architecture or internal knowledge. More importantly, we take recent advances in randomized smoothing theory and propose a novel weightbased distribution algorithm to enable semantic similarity and provide theoretical robustness guarantees. Experimentally, we demonstrate the efficacy of our approach across a diverse range of datasets and tasks, highlighting its utility in mitigating backdoor triggers. Our results show strong performance in terms of certified accuracy, scalability, and semantic preservation.

## Overview
<img src="figs/overview.png" width=1000></img>

## Data Preparation

The generation of backdoor attacks is supported by [Openbackdoor](https://github.com/thunlp/OpenBackdoor). More details can be found in [Openbackdoor](https://github.com/thunlp/OpenBackdoor).

Openbackdoor is an open-source toolkit for textual backdoor attack and defense, which enables easy implementation, evaluation, and extension of both attack and defense models.
The below code blocks present examples of built-in backdoor attack. 

### Attack

```python
# Attack BERT on SST-2 with BadNet
import openbackdoor as ob 
from openbackdoor import load_dataset
# choose BERT as victim model 
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")
# choose BadNet attacker
attacker = ob.Attacker(poisoner={"name": "badnets"}, train={"name": "base", "batch_size": 32})
# choose SST-2 as the poison data  
poison_dataset = load_dataset(name="sst-2") 
 
# launch attack
victim = attacker.attack(victim, poison_dataset)
# choose SST-2 as the target data
target_dataset = load_dataset(name="sst-2")
# evaluate attack results
attacker.eval(victim, target_dataset)
```

## How to Run

### Data Preprocess
```python
python data_process.py
```
### Train 
```python
python train_crowd.py
```

### Evaluation
```python
python eval_crowd.py
```

## Citation
If you find our method useful, please kindly cite our paper:
```
@inproceedings{sun-etal-2024-crowd,
    title = "{CROWD}: Certified Robustness via Weight Distribution for Smoothed Classifiers against Backdoor Attack",
    author = "Sun, Siqi  and
      Sen, Procheta  and
      Ruan, Wenjie",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.993",
    doi = "10.18653/v1/2024.findings-emnlp.993",
    pages = "17056--17070",
}
```
