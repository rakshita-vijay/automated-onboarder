import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

SENT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TRAIN_DATA_DIR = "training_data"
TRAIN_RESUMES_DIR = os.path.join(TRAIN_DATA_DIR, "training_resumes")
os.makedirs(TRAIN_RESUMES_DIR, exist_ok=True)

class CrossCheckerModel:
  def __init__(self, csv_path=None, emb_model=SENT_EMB_MODEL):
    if csv_path is None:
      csv_path = os.path.join(TRAIN_RESUMES_DIR, "resume_final.csv")
    try:
      self.df = pd.read_csv(csv_path)
    except Exception as e:
      print(f"Warning: CSV load failed: {e}. Creating empty dataframe.")
      self.df = pd.DataFrame(columns=['name', 'completeness', 'truthiness'])

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
    sim = (sim + 1) / 2
    return round(100 * max(0, min(1, sim)), 2)

  def get_scores(self, applicant_name, resume_text, jd_text):
    # Clean applicant name for matching
    clean_name = applicant_name.replace("_", " ").lower()
    row = self.df[self.df['name'].str.lower() == clean_name]
    completeness = float(row['completeness'].iloc[0]) if not row.empty else 50.0
    truthiness = float(row['truthiness'].iloc[0]) if not row.empty else 50.0
    relevance = self.compute_relevance(resume_text, jd_text)
    return {'completeness': completeness, 'truthiness': truthiness, 'relevance': relevance}
