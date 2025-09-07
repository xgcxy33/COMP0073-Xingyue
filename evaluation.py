#All code logic was fully understood, and AI was only used in an assistive role. All methods, analysis, and logical reasoning were developed independently by the author. This usage falls under Category 2 of UCL’s Generative AI policy: assistive use only.
import argparse
import json
import csv
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

# ==================== Metric 1: BERTScore ====================
def compute_bertscore(text_a, text_b):
    P, R, F1 = bert_score([text_a], [text_b], lang="en")
    return P.mean().item(), R.mean().item(), F1.mean().item()

# ==================== Metric 2: Bio_ClinicalBERT + Cosine ====================
class BioTextSimilarity:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])

        return np.array(embeddings)

    def compute_similarity(self, text1, text2):
        embeddings = self.get_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

# ==================== Metric 3: Sentence-BERT + Cosine ====================
def compute_sentencebert_similarity(text_a, text_b):
    print("=== Sentence-BERT Cosine Similarity ===")
    device = 'cpu'  
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embedding_a = model.encode([text_a], convert_to_tensor=True, device=device)
    embedding_b = model.encode([text_b], convert_to_tensor=True, device=device)
    similarity = cosine_similarity(embedding_a.cpu().numpy(), embedding_b.cpu().numpy())
    return similarity[0][0]

# ==================== Metric 4: ROUGE-1 ====================
def compute_rouge(reference, generated):
    print("=== ROUGE-1 Score ===")
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    rouge = scores["rouge1"]
    return rouge.precision, rouge.recall, rouge.fmeasure


def main(args):
    with open(args.input_file_path, 'r') as f:
        data = json.load(f)

    output_csv_path = "results.csv"
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "image_path",
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
            "bio_bert_cosine",
            "sentence_bert_cosine",
            "rouge1_precision",
            "rouge1_recall",
            "rouge1_f1",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        bio_model = BioTextSimilarity()

        for row in data["outputs"]:
            print(f"\n=== Processing: {row['image_path']} ===")
            text_a = row["literature"]
            text_b = row["clinic_output"]

            # Compute scores
            b_precision, b_recall, b_f1 = compute_bertscore(text_a, text_b)
            bio_sim = bio_model.compute_similarity(text_a, text_b)
            sentence_sim = compute_sentencebert_similarity(text_a, text_b)
            r_precision, r_recall, r_f1 = compute_rouge(text_a, text_b)

            # Print results
            print(f"BERTScore - Precision: {b_precision:.4f}, Recall: {b_recall:.4f}, F1: {b_f1:.4f}")
            print(f"Bio_ClinicalBERT Cosine: {bio_sim:.4f}")
            print(f"Sentence-BERT Cosine: {sentence_sim:.4f}")
            print(f"ROUGE-1 - Precision: {r_precision:.4f}, Recall: {r_recall:.4f}, F1: {r_f1:.4f}")

            # Write to CSV
            writer.writerow({
                "image_path": row["image_path"],
                "bertscore_precision": b_precision,
                "bertscore_recall": b_recall,
                "bertscore_f1": b_f1,
                "bio_bert_cosine": bio_sim,
                "sentence_bert_cosine": sentence_sim,
                "rouge1_precision": r_precision,
                "rouge1_recall": r_recall,
                "rouge1_f1": r_f1,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path", help="Path to the input JSON file")
    args = parser.parse_args()
    main(args)
