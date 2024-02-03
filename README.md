# DoGE
| Codebase for ICML submission "DOGE: Domain Reweighting with Generalization Estimation"

## Requirements
> pip install -r requirements.txt

## Run DoGE-proxy for Domain Reweighting
- Universal Generalization on SlimPajama
replace `WANDB_API_KEY` by your own authorize key. 
> bash script/doge.sh

- Out-of-Domain Generalization
**SlimPajama**
> bash script/doge_ood.sh

**Wiki40b-Catalan**
> bash script/doge_ood_wiki40b.sh

## Train Base Model (DoGE-base)
> bash script/base.sh