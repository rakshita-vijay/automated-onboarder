import os
import pandas as pd
import numpy as np
import requests
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By

def init_driver():
  options = ChromeOptions()
  options.add_argument("--headless")
  options.add_argument("--no-sandbox")
  options.add_argument("--disable-dev-shm-usage")
  driver = webdriver.Chrome(options=options)
  return driver

def search_links(query, driver):
  driver.get(f"https://www.google.com/search?q={query.replace(' ', '+')}")
  time.sleep(2.0)
  links = []
  try:
      elements = driver.find_elements(By.XPATH, "//a[@href]")
    for elem in elements:
      link = elem.get_attribute("href")
      if any(s in link for s in ['github.com', 'linkedin.com', 'facebook.com', 'leetcode.com']):
        links.append(link)
  except Exception:
    pass
  return list(set(links))

def fetch_url_for_person(url, name):
  try:
    page = requests.get(url, timeout=10).text.lower()
    return name.lower() in page
  except:
    return False

def extract_features(text):
  # Basic extraction: project names, features, contributions
  proj_matches = re.findall(r"project[:\- ](.*?)(?=\.|\n|;|$)", text, flags=re.IGNORECASE)
  feature_matches = re.findall(r"(feature[s]*[:\- ](.*?)(?=\.|\n|;|$))", text, flags=re.IGNORECASE)
  return proj_matches, feature_matches

class TruthinessModel:
  """
  Model to assign 'truthiness' scores to resume entries by evaluating their factual credibility.
  - Loads the combined resume (with completeness column) and JD datasets.
  - Fine-tunes a BERT sequence pair classifier for NLI (entailment) on resume x JD/text evidence pairs.
  - Adds a 'truthiness' column to 'resume_final.csv'.
  - Model trained once at startup and saved for inference.
  """

  def __init__(self):
    self.base_dir = "training_data"
    self.resume_dir = os.path.join(self.base_dir, "training_resumes")
    self.jd_dir = os.path.join(self.base_dir, "training_jds")
    self.resume_infiles = [
      os.path.join(self.resume_dir, f"resume_final_{i+1}.csv") for i in range(3)
    ]
    self.jd_infiles = [
      os.path.join(self.jd_dir, f"jd_final_{i+1}.csv") for i in range(3)
    ]

    self.model_dir = "models/saved"
    self.model_path = os.path.join(self.model_dir, "truthiness_bert.pt")
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    os.makedirs(self.model_dir, exist_ok=True)

  def load_aggregated_datasets(self):
    """
    Loads resumes with completeness column and JD text as DataFrames.
    """
    # resume_df = pd.read_csv(os.path.join(self.base_dir, self.resume_infile))

    for infile in self.resume_infiles:
      resume_df = pd.read_csv(infile)
      # ... process as before ...
      resume_df.to_csv(infile, index=False)

    jd_df = pd.read_csv(os.path.join(self.base_dir, self.jd_infile))
    resume_df = resume_df.dropna(subset=["text"])
    jd_df = jd_df.dropna(subset=["text"])
    return resume_df, jd_df

  def create_resume_jd_pairs(self, resume_df, jd_df, max_pairs=2500):
    """
    For the NLI setup, pairs each resume statement with a random JD description.
    """
    pairs = []

    # For demonstration, match each resume to a random JD as negative example + a positive self-pair
    for ix, row in resume_df.iterrows():
      # The premise: what we want to check (resume), the hypothesis: job description evidence
      # We label "true" (1) for direct matching, "false" (0) otherwise.
      # In practical NLI, you should scrape true web evidence, but for this automodel, we use JD as a proxy.
      pairs.append({
        "premise": str(row["text"]),
        "hypothesis": str(jd_df.sample(1)["text"].values[0]),
        "label": 0
      })

      # Add positive (inferred true) pairs: treat "complete" resumes as also "truthy" for self-supervision
      if row.get("completeness", 0) == 1:
        pairs.append({
          "premise": str(row["text"]),
          "hypothesis": str(row["text"]),
          "label": 1
        })

      if len(pairs) >= max_pairs:
        break
    pairs_df = pd.DataFrame(pairs)
    return pairs_df

  def tokenize_pairs(self, premises, hypotheses):
    """
    Tokenizes the premise-hypothesis pairs for BERT-for-NLI.
    """
    return self.tokenizer(
      list(premises), list(hypotheses),
      padding=True, truncation=True, max_length=256, return_tensors="pt"
    )


def run(self):
  for resume_file in self.resume_infiles:
    if not os.path.exists(resume_file):
      continue

    # Load each resume file and process with all JD files
    resume_df = pd.read_csv(resume_file)

    # Combine all JD data for training
    all_jd_data = []
    for jd_file in self.jd_infiles:
      if os.path.exists(jd_file):
        jd_df = pd.read_csv(jd_file)
        all_jd_data.append(jd_df)

    if not all_jd_data:
      st.warning("[WARN] No JD data found, skipping truthiness model.")
      continue

    combined_jd_df = pd.concat(all_jd_data, ignore_index=True)

    if resume_df.empty or combined_jd_df.empty:
      st.warning(f"[WARN] Resume or JD data empty for {resume_file}, skipping.")
      continue

    pairs_df = self.create_resume_jd_pairs(resume_df, combined_jd_df)

    # Continue with existing training logic...
    X_train, X_eval, y_train, y_eval = train_test_split(
      pairs_df[["premise", "hypothesis"]], pairs_df["label"],
      test_size=0.2, random_state=42
    )

    # Rest of the training code stays the same...
    train_encodings = self.tokenize_pairs(X_train["premise"], X_train["hypothesis"])
    eval_encodings = self.tokenize_pairs(X_eval["premise"], X_eval["hypothesis"])
    train_labels = torch.tensor(list(y_train))
    eval_labels = torch.tensor(list(y_eval))

    class NLIDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
      def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
      def __len__(self):
        return len(self.labels)

    train_dataset = NLIDataset(train_encodings, train_labels)
    eval_dataset = NLIDataset(eval_encodings, eval_labels)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    training_args = TrainingArguments(
      output_dir=self.model_dir,
      do_train=True,
      do_eval=True,
      num_train_epochs=1,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      warmup_steps=10,
      weight_decay=0.01,
      logging_dir='./logs',
      logging_steps=20,
      save_strategy="no"
    )

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
    )

    with st.spinner("Training TruthinessModel on resume and JD pairings..."):
      trainer.train()
    model.save_pretrained(self.model_dir)
    self.tokenizer.save_pretrained(self.model_dir)
    torch.save(model.state_dict(), self.model_path)

    # Web-based truthiness scoring
    driver = init_driver()
    truthiness_scores = []
    for _, row in resume_df.iterrows():
      try:
        name = row.get("name", "")
        text = str(row["text"])
        projects, features = extract_features(text)
        found = 0
        total = 0
        to_check = list(projects) + list(features)
        for entry in to_check:
          for site in ["github", "linkedin", "leetcode", "facebook"]:
            result_links = search_links(f"{entry} {name} {site}", driver)
            for link in result_links:
              if fetch_url_for_person(link, name):
                found += 1
                break
            total += 1
        score = int((found / total) * 100) if total > 0 else 0
        truthiness_scores.append(score)
      except Exception:
        truthiness_scores.append(0)
    driver.quit()

    resume_df["truthiness"] = truthiness_scores
    resume_df.to_csv(resume_file, index=False)

  st.info(f"[INFO] Truthiness model trained and applied. Model saved at {self.model_path}")

# Support function for application_evaluator.py workflow
def create_initial_dataset():
  """
  Helper for app startup protection. No-op since base CSV is already created by CompletenessModel.
  """
  pass

# For manual test
if __name__ == "__main__":
  TruthinessModel().run()

prev = '''import os
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
'''
