import os
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline
import requests

FACT_CHECK_MODEL = "microsoft/deberta-v3-base-mnli"  # HuggingFace for entailment/contradiction

def fetch_url_text(url, name):
  """Download url and check if the claimed person is really present."""
  try:
    resp = requests.get(url, timeout=5)
    content = resp.text.lower()
    if name and name.lower() in content:
      return True
  except Exception:
    pass
  return False

def extract_projects_features(text):
  # Naive split: look for 'project: ... features: ...'
  segments = re.split(r"project[s]*[:\-]", text, flags=re.IGNORECASE)
  facets = []
  for seg in segments[1:]:
    m = re.search(r"(feature[s]*[:\-])(.*?)([.\n])", seg)
    if m:
      facets.append(m.group(2).strip())
  return facets

class TruthinessModel:
  def __init__(self, csv_in="resume_augmented.csv", csv_out="resume_final.csv", entail_model=FACT_CHECK_MODEL):
    self.csv_in = csv_in
    self.csv_out = csv_out
    self.df = pd.read_csv(self.csv_in)
    self.fact_pipe = pipeline("zero-shot-classification", model=entail_model)

  def row_score(self, row):
    name = row.get("name") or extract_name_ner(row.get("text",""))
    links = str(row.get("links",""))
    # Check links
    truth = []
    for url in links.split(","):
      if "http" in url and fetch_url_text(url, name):
        truth.append(1.)
      else:
        truth.append(0.)
    proj = row.get("projects", "")
    # If project features are listed, check them using entailment
    facets = extract_projects_features(str(proj))
    facet_scores = []
    for facet in facets:
      res = self.fact_pipe(facet, [proj])
      score = float(res["scores"][0]) if "scores" in res else 0
      facet_scores.append(score)
    # Aggregate
    if truth:
      sc1 = max(truth)
    else:
      sc1 = 0.
    if facet_scores: sc2 = sum(facet_scores)/len(facet_scores)
    else: sc2 = 0.
    return min(100, round((0.6*sc1+0.4*sc2)*100, 2))

  def run(self):
    self.df["truthiness"] = 0.
    for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
      try:
        self.df.at[i, "truthiness"] = self.row_score(row)
      except Exception:
        self.df.at[i, "truthiness"] = 0
    self.df.to_csv(self.csv_out, index=False)
    print("Truthiness scores added to", self.csv_out)

if __name__ == "__main__":
  TruthinessModel().run()
