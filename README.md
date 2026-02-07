# TOPSIS for Text Sentence Similarity

This project applies the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** method to select the best **pre-trained text sentence similarity model** using multiple evaluation criteria.

---

## Objective

To rank pre-trained sentence similarity models by balancing accuracy, efficiency, and resource usage using the TOPSIS multi-criteria decision-making approach.

---

## Models Evaluated

- Sentence-BERT (all-MiniLM-L6-v2)
- Sentence-BERT (all-mpnet-base-v2)
- Universal Sentence Encoder (USE)
- DistilBERT (STS fine-tuned)
- RoBERTa (STS fine-tuned)

---

## Evaluation Criteria

| Criterion | Impact |
|---------|--------|
| STS Score | + |
| Inference Time (ms) | - |
| Model Size (MB) | - |
| Embedding Dimension | + |

---

## Dataset

The decision matrix is stored in:
data/sentence_similarity_models.csv

Each row represents a model and each column represents an evaluation criterion.

---

## Methodology

1. Normalize the decision matrix using vector normalization  
2. Apply weights to normalized criteria  
3. Determine positive and negative ideal solutions  
4. Compute distances from ideal solutions  
5. Calculate TOPSIS score and rank models

---

## Results

Based on TOPSIS analysis, DistilBERT achieved the highest rank, indicating the best overall trade-off between accuracy, speed, and model size among the evaluated models.
