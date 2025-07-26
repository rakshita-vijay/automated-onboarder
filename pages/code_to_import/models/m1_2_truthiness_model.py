import os
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline
import requests
import re

FACT_CHECK_MODEL = "microsoft/deberta-v3-base-mnli"

TRAIN_RESUMES_DIR = "training_data/training_resumes"
os.makedirs(TRAIN_RESUMES_DIR, exist_ok=True)

# Change all csv_in/csv_out to use TRAIN_RESUMES_DIR, e.g.:
self.csv_in = os.path.join(TRAIN_RESUMES_DIR, "resume_augmented.csv")
self.csv_out = os.path.join(TRAIN_RESUMES_DIR, "resume_final.csv")

def setup_git_repo():
  try:
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    repo = git.Repo(".")
    username = "rakshita-vijay"
    repo_url = f"https://{username}:{GITHUB_TOKEN}@github.com/{username}/automated-onboarder.git"
    repo.remote().set_url(repo_url)
    st.success("Using existing Git repository!")
    return repo
  except git.exc.InvalidGitRepositoryError:
    st.error("Not in a Git repository. Make sure you're running from your repo directory.")
    return None

def extract_name_ner(text):
  from transformers import pipeline
  ner_pipe = pipeline("ner", model=NER_MODEL, grouped_entities=True)
  ent = ner_pipe(text[:400])
  for e in ent:
    if e['entity_group'] == "PER":
      return e['word']
  return ""

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
  segments = re.split(r"project[s]*[:\-]", text, flags=re.IGNORECASE)
  facets = []
  for seg in segments[1:]:
    m = re.search(r"(feature[s]*[:\-])(.*?)([.\n])", seg)
    if m:
      facets.append(m.group(2).strip())
  return facets

def check_github_contributions(url, name, facets):
  """If GitHub URL, check commits for name and if facets (features) appear in logs."""
  if "github.com" not in url: return 0.5
  try:
    api_url = url.replace("github.com", "api.github.com/repos") + "/commits"
    resp = requests.get(api_url, timeout=5)
    commits = resp.json()
    contrib = any(name.lower() in str(c.get("commit", {}).get("author", {})) for c in commits)
    facet_match = any(any(f.lower() in str(c.get("commit", {}).get("message", "")) for f in facets) for c in commits)
    return 1.0 if contrib and facet_match else 0.7 if contrib else 0.3
  except:
    return 0.3

class TruthinessModel:
  def __init__(self, csv_in="resume_augmented.csv", csv_out="resume_final.csv", entail_model=FACT_CHECK_MODEL):
    self.csv_in = csv_in
    self.csv_out = csv_out
    self.df = pd.read_csv(self.csv_in)
    self.fact_pipe = pipeline("zero-shot-classification", model=entail_model)

  def row_score(self, row):
    name = row.get("name") or extract_name_ner(row.get("text",""))
    links = str(row.get("links",""))
    truth = []

    for url in links.split(","):
      if "http" in url and fetch_url_text(url, name):
        truth.append(1.)
      else:
        truth.append(0.)

    proj = row.get("projects", "")

    if not proj:
      import re
      txt = str(row.get("text", ""))
      proj_match = re.search(r"project[s]*[:\-](.*?)(?=(\n\n|\Z))", txt, re.DOTALL | re.IGNORECASE)
      proj = proj_match.group(1).strip() if proj_match else ""

    facets = extract_projects_features(str(proj))
    facet_scores = []

    for facet in facets:
      res = self.fact_pipe(facet, [proj])
      score = float(res["scores"][0]) if "scores" in res else 0
      facet_scores.append(score)

    gh_score = 0
    gh_links = [l for l in links.split(",") if "github.com" in l]

    if gh_links:
      gh_score = check_github_contributions(gh_links[0], name, facets)

    if truth:
      sc1 = max(truth)
    else:
      sc1 = 0.

    if facet_scores:
      sc2 = sum(facet_scores)/len(facet_scores)
    else:
      sc2 = 0.

    sc3 = gh_score
    return min(100, round((0.4*sc1 + 0.3*sc2 + 0.3*sc3)*100, 2))

  def run(self):
    if os.path.exists(self.csv_out):
      print(f"{self.csv_out} exists; skipping recompute.")
      return

    self.df["truthiness"] = 0.

    for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
      try:
        self.df.at[i, "truthiness"] = self.row_score(row)
      except Exception:
        self.df.at[i, "truthiness"] = 0

    self.df.to_csv(self.csv_out, index=False)


    # Train regressors on embeddings for fast prediction
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import joblib
    from transformers import AutoTokenizer, AutoModel
    import torch

    SENT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    tokenizer = AutoTokenizer.from_pretrained(SENT_EMB_MODEL)
    model = AutoModel.from_pretrained(SENT_EMB_MODEL)

    @torch.no_grad()
    def get_emb(text):
      inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=384)
      emb = model(**inputs).last_hidden_state[:,0,:].squeeze().cpu()
      return emb

    embeddings = []
    completeness_labels = []
    truthiness_labels = []

    for _, row in self.df.iterrows():
      text = str(row['text'])
      emb = get_emb(text)
      embeddings.append(emb.numpy())
      completeness_labels.append(row['completeness'])
      truthiness_labels.append(row['truthiness'])

    X = np.stack(embeddings)

    comp_model = LinearRegression().fit(X, completeness_labels)
    truth_model = LinearRegression().fit(X, truthiness_labels)

    joblib.dump(comp_model, os.path.join(TRAIN_RESUMES_DIR, 'completeness_regressor.joblib'))
    joblib.dump(truth_model, os.path.join(TRAIN_RESUMES_DIR, 'truthiness_regressor.joblib'))

    repo = setup_git_repo()
    if repo:
      repo.git.add(self.csv_out)  # Add only the final CSV
      repo.index.commit("Update resume_final.csv with truthiness scores")
      origin = repo.remote(name="origin")
      origin.push()
      print("Pushed resume_final.csv to GitHub")

    print("Truthiness scores added to", self.csv_out)

if __name__ == "__main__":
  TruthinessModel().run()
