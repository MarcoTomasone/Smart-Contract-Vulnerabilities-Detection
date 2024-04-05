from datasets import load_dataset

val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', ignore_verifications=True)
print(val_set[2])