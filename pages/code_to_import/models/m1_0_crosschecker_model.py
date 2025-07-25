import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

SENT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class CrossCheckerModel:
  def __init__(self, csv_path="resume_final.csv", emb_model=SENT_EMB_MODEL):
    self.df = pd.read_csv(csv_path)
    self.tokenizer = AutoTokenizer.from_pretrained(emb_model)
    self.model = AutoModel.from_pretrained(emb_model)

  @torch.no_grad()
  def get_embedding(self, text):
    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=384)
    emb = self.model(**inputs).last_hidden_state[:,0,:].squeeze().cpu()
    return emb

  def compute_relevance(self, resume_text, jd_text):
    emb1 = self.get_embedding(resume_text)
    emb2 = self.get_embedding(jd_text)
    sim = torch.cosine_similarity(emb1, emb2, dim=0).item()
    return round(100 * max(0, min(1, sim)), 2)

  def get_scores(self, applicant_name, resume_text, jd_text):
    row = self.df[self.df['name'].str.lower() == applicant_name.lower()]
    completeness = float(row['completeness'].iloc[0]) if not row.empty else None
    truthiness = float(row['truthiness'].iloc[0]) if not row.empty else None
    relevance = self.compute_relevance(resume_text, jd_text)
    return {'completeness': completeness, 'truthiness': truthiness, 'relevance': relevance}

# Usage in your app:
# crosschecker = CrossCheckerModel()
# scores = crosschecker.get_scores(applicant_name, resume_text, jd_text)
