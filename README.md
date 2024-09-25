# FdFTL
The implementation of "Federated Fuzzy Transfer Learning With Domain and Category Shifts" in Python. 

Code for the TFS publication. The full paper can be found [here](https://doi.org/10.1109/TFUZZ.2024.3459927). 

Plese manually download datasets from official websites, including Office-31, Office-Home, DomainNet.

## Usage
### OPDA
bash ./scripts/train_offh_fuz_OPDA.sh

bash ./scripts/target_offh_fuz_OPDA.sh 

bash ./scripts/train_off31_fuz_OPDA.sh

bash ./scripts/target_off31_fuz_OPDA.sh 

bash ./scripts/train_dnet_fuz_OPDA.sh

bash ./scripts/target_dnet_fuz_OPDA.sh 

### OSDA:
bash ./scripts/train_source_fuz_OSDA.sh

bash ./scripts/train_target__fuz_OSDA.sh 

### PDA:
bash ./scripts/train_source_fuz_PDA.sh

bash ./scripts/train_target__fuz_PDA.sh 

### CLDA:
bash ./scripts/train_source_fuz_CLDA.sh

bash ./scripts/train_target__fuz_CLDA.sh 


Please consider citing if you find this helpful or use this code for your research.

Citation
```
@article{li2024federated,
  title={Federated Fuzzy Transfer Learning With Domain and Category Shifts},
  author={Li, Keqiuyin and Lu, Jie and Zuo, Hua and Zhang, Guangquan},
  journal={IEEE Transactions on Fuzzy Systems},
  year={2024},
  publisher={IEEE}
}
