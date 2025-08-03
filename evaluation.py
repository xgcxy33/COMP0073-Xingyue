import argparse
import json
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

# ==================== 方法 1: BERTScore ====================
def compute_bertscore(text_a, text_b):
    print("=== BERTScore ===")
    P, R, F1 = bert_score([text_a], [text_b], lang="en")
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall:    {R.mean().item():.4f}")
    print(f"F1 Score:  {F1.mean().item():.4f}")

# ==================== 方法 2: Bio_ClinicalBERT + Cosine ====================
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

# ==================== 方法 3: Sentence-BERT + Cosine ====================
def compute_sentencebert_similarity(text_a, text_b):
    print("=== Sentence-BERT Cosine Similarity ===")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # or use 'all-mpnet-base-v2' for better accuracy
    embedding_a = model.encode([text_a])
    embedding_b = model.encode([text_b])
    similarity = cosine_similarity(embedding_a, embedding_b)
    print(f"Cosine similarity: {similarity[0][0]:.4f}")

# ==================== 方法 4: ROUGE-1 ====================
def compute_rouge(reference, generated):
    print("=== ROUGE-1 Score ===")
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    print(f"ROUGE-1 Precision: {scores['rouge1'].precision:.4f}")
    print(f"ROUGE-1 Recall:    {scores['rouge1'].recall:.4f}")
    print(f"ROUGE-1 F1 Score:  {scores['rouge1'].fmeasure:.4f}")

# ==================== 主程序入口 ====================
def main(args):
    with open(args.input_file_path, 'r') as f:
        data = json.load(f)

    for row in data['outputs']:
        print(f"=== Processing record: {row['image_path']} ===")
        text_a = row["literature"]
        text_b = row["clinic_output"]

        compute_bertscore(text_a, text_b)

        print("\n=== Bio_ClinicalBERT Cosine Similarity ===")
        bio_model = BioTextSimilarity()
        bio_sim = bio_model.compute_similarity(text_a, text_b)
        print(f"Cosine similarity: {bio_sim:.4f}")

        print()
        compute_sentencebert_similarity(text_a, text_b)

        print()
        compute_rouge(text_a, text_b)

# ==================== 执行 ====================
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("input_file_path", help="Input file path")

    # Parse the arguments
    args = parser.parse_args()
    main(args)
