# Multilingual News Article Similarity

IIIT-MLNS at SemEval-2022 shared task # 8.

Base architecture: Siamese network

## Experiments performed:

1. Features: News text, metadata, NER-extracted features
2. Base encoder model: XLM-RoBERTa, Multilingual DistilBERT
3. Encoder representation concatenations: \[|x1 - x2|; (x1 + x2)/2\], \[x1; x2; |x1 - x2|\]
4. Data augmentation
5. Output activation
6. Ensembling
