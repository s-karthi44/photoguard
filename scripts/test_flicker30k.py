from datasets import load_dataset

dataset = load_dataset("AminDehnavi/flickr30k")

print(dataset)
print(dataset["train"][0])