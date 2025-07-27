import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np

class CrossCheckerModel:
  """
  - Loads the final resume dataset after completeness & truthiness have been added, as well as an up-to-date JD data file.
  - Trains/fine-tunes a BERT model to return a 'relevance' score (resume x JD semantic similarity/classification).
  - Calculates and returns completeness and truthiness for a given applicant by looking up this final dataset.
  - Pushing the resume dataset to GitHub happens only at startup/reboot; no accidental overwrite mid-session.
  - The model is saved and NOT re-trained except on app reboot.
  - get_scores provides the API for Streamlit integration in evaluation page.
  """

  def __init__(self):
    self.base_dir = "training_data"
    self.resume_dir = os.path.join(self.base_dir, "training_resumes")
    self.resume_outfiles = [
      os.path.join(self.resume_dir, f"resume_final_{i+1}.csv") for i in range(3)
    ]
    os.makedirs(self.resume_dir, exist_ok=True)

    self.jd_dir = os.path.join(self.base_dir, "training_jds")
    self.jd_outfiles = [
      os.path.join(self.jd_dir, f"jd_final_{i+1}.csv") for i in range(3)
    ]
    os.makedirs(self.jd_dir, exist_ok=True)

    self.model_dir = "models/saved"
    self.model_path = os.path.join(self.model_dir, "crosschecker_bert.pt")
    self.tokenizer_path = os.path.join(self.model_dir, "crosschecker_tokenizer")
    self.epochs = 2        # Set to 2 for better learning than the default 1
    self.score_cache = {}  # For fast lookup after training
    os.makedirs(self.model_dir, exist_ok=True)

    # Setup tokenizer and model
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Model for pair classification: 2 labels (match, not match/relevant)
    self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = self.model.to(self.device)

    # Keep track of whether data/model has already been pushed to GitHub in this session
    self.resume_file_pushed = False

  def push_resume_file_to_github(self):
    """
    Pushes the final, edited resume dataset to GitHub ONCE per reboot, never more.
    Uses the pattern shown in your p1_upload_resume.py, with appropriate adjustments.
    """
    # if self.resume_file_pushed or not os.path.exists(os.path.join(self.base_dir, self.resume_file)):
    if self.resume_file_pushed or not os.path.exists(self.resume_file):
      return
    try:
      import git
      GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
      repo = git.Repo(".")
      username = "rakshita-vijay"
      repo_url = f"https://{username}:{GITHUB_TOKEN}@github.com/{username}/automated-onboarder.git"
      repo.remote().set_url(repo_url)
      # Only add and commit the CSV file, not JDs or models
      resume_file_path = self.resume_file  # Use the defined primary file
      repo.git.add(resume_file_path)
      repo.index.commit("Update primary resume_final.csv with completeness, truthiness, and relevance scores")

      origin = repo.remote(name="origin")
      origin.push()
      self.resume_file_pushed = True
      st.info("[INFO] resume_final.csv pushed to GitHub.")
    except Exception as e:
      st.warning(f"[WARN] Push to GitHub failed: {e}. Resume file NOT pushed.")

  def load_data(self):
    """
    Loads the final resume and JD datasets for model training and lookup.
    Resume data must already have 'completeness' and 'truthiness' columns added.
    JD data can be downloaded anew each run.
    """

    self.resume_dir = os.path.join(self.base_dir, "training_resumes")
    self.jd_dir = os.path.join(self.base_dir, "training_jds")
    self.resume_files = sorted([
      os.path.join(self.resume_dir, f) for f in os.listdir(self.resume_dir) if f.startswith("resume_final_")
    ])
    self.jd_files = sorted([
      os.path.join(self.jd_dir, f) for f in os.listdir(self.jd_dir) if f.startswith("jd_final_")
    ])

    # Define a primary resume file for GitHub push (e.g., the first one)
    if self.resume_files:
      self.resume_file = self.resume_files[0]  # Or choose based on logic, e.g., the most recent
    else:
      self.resume_file = os.path.join(self.resume_dir, "resume_final_1.csv")  # Fallback

    if not self.resume_files:
      st.warning("[WARN] No resume training files found. Using empty dataframe for resumes.")
    if not self.jd_files:
      st.warning("[WARN] No JD training files found. Using empty dataframe for JDs.")

    resume_dfs = [pd.read_csv(f) for f in self.resume_files]
    jd_dfs = [pd.read_csv(f) for f in self.jd_files]

    # Load and combine all resume and JD dataframes
    resume_dfs = [pd.read_csv(f) for f in self.resume_files if os.path.exists(f)]
    jd_dfs = [pd.read_csv(f) for f in self.jd_files if os.path.exists(f)]

    all_resume_dfs = [df.dropna(subset=["text"]) for df in resume_dfs]
    all_jd_dfs = [df.dropna(subset=["text"]) for df in jd_dfs]

    combined_resume_df = pd.concat(all_resume_dfs, ignore_index=True) if all_resume_dfs else pd.DataFrame()
    combined_jd_df = pd.concat(all_jd_dfs, ignore_index=True) if all_jd_dfs else pd.DataFrame()
    return combined_resume_df, combined_jd_df

  def create_pair_dataset(self, resume_df, jd_df, n_per_resume=3):
    """
    For each resume entry, pairs it with n random JDs as positive/negative relevance pairs.
    Positive: the resume vs. a JD that's likely to match its own area (rough simulation).
    For now, all pairings are treated as potential positive with a similarity measure, but label randomly for demo.
    """
    # Aggregate random pairs
    pairs = []
    for resume_file in self.resume_files:
      resume_df = pd.read_csv(resume_file).dropna(subset=["text"])
      for jd_file in self.jd_files:
        jd_df = pd.read_csv(jd_file).dropna(subset=["text"])
        # For each resume row, create N random pairs with JDs from THIS jd_df
        for _, r_row in resume_df.iterrows():
          resume_text = str(r_row["text"])
          # Number of random pairs per resume (adjustable)
          N_PAIRS = 3
          sample_jd_texts = jd_df["text"].sample(n=min(N_PAIRS, len(jd_df)), replace=False, random_state=None).tolist()
          for jd_text in sample_jd_texts:
            # Random label logic or leave as 0 for now; you could randomize, or use 1 for "synthetic positive" if criteria matches
            label = 0
            pairs.append({
              "resume": resume_text,
              "jd": jd_text,
              "label": label
            })
          # Optionally, add an extra positive pair (same resume with itself or with a randomly matched JD)
          pos_jd_text = jd_df["text"].sample(1, random_state=None).iloc[0]
          pairs.append({
            "resume": resume_text,
            "jd": pos_jd_text,
            "label": 1
          })
          if len(pairs) > 3200:  # keep final dataset manageable
            break

    # Convert pairs to DataFrame for further processing, deduplication, etc.
    pairs_df = pd.DataFrame(pairs)

    return pairs_df

  def tokenize_pair_batch(self, resumes, jds):
    """
    Tokenize batches of resume/JD string pairs for BERT sequence-pair input.
    """
    return self.tokenizer(list(resumes), list(jds), padding=True, truncation=True, max_length=256, return_tensors="pt")

  def run(self):
    """
    Main entrypoint. Triggers at app reboot/startup:
    - Loads and/or prepares data, including checking completeness/truthiness columns.
    - Pushes edited resume_final.csv to GitHub, ONCE.
    - Trains relevance model (resume x JD semantic/classification) and saves it to models/saved/.
    - Pre-caches all predicted scores for fast lookup at inference time.
    """
    resume_df, jd_df = self.load_data()
    self.push_resume_file_to_github()  # GitHub push logic

    # Build training data from resume/JD random pairs
    pairs_df = self.create_pair_dataset(resume_df, jd_df)
    X_train, X_eval, y_train, y_eval = train_test_split(
      pairs_df[["resume", "jd"]], pairs_df["label"],
      test_size=0.2, random_state=42
    )
    train_encodings = self.tokenize_pair_batch(X_train["resume"], X_train["jd"])
    eval_encodings = self.tokenize_pair_batch(X_eval["resume"], X_eval["jd"])
    train_labels = torch.tensor(list(y_train))
    eval_labels = torch.tensor(list(y_eval))

    class RelevancePairDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
      def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
      def __len__(self):
        return len(self.labels)

    train_dataset = RelevancePairDataset(train_encodings, train_labels)
    eval_dataset = RelevancePairDataset(eval_encodings, eval_labels)

    training_args = TrainingArguments(
      output_dir=self.model_dir,
      do_train=True,
      do_eval=True,
      num_train_epochs=self.epochs,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      warmup_steps=10,
      weight_decay=0.01,
      logging_dir='./logs',
      logging_steps=20,
      save_strategy="no"
    )

    trainer = Trainer(
      model=self.model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
    )
    st.info("[INFO] Training CrossCheckerModel on resume x JD pairings...")
    trainer.train()
    self.model.save_pretrained(self.model_dir)
    self.tokenizer.save_pretrained(self.tokenizer_path)
    torch.save(self.model.state_dict(), self.model_path)

    st.info("[INFO] Precomputing all resume/JD pairs for ultra-fast lookup...")
    self.score_cache = {}
    for ix, r_row in resume_df.iterrows():
      applicant_name = r_row.get("name", f"applicant_{ix}") if "name" in r_row else f"applicant_{ix}"
      resume_text = r_row["text"]
      completeness = float(r_row.get("completeness", 0))
      truthiness = float(r_row.get("truthiness", 0))
      for jx, j_row in jd_df.iterrows():
        jd_text = j_row["text"]
        key = (applicant_name, jd_text[:80])  # longer JD key for accuracy
        relevance = self.compute_relevance(resume_text, jd_text)
        self.score_cache[key] = {
          "completeness": completeness * 100 if completeness < 1.01 else completeness,
          "truthiness": truthiness * 100 if truthiness < 1.01 else truthiness,
          "relevance": relevance * 100
        }
    st.info(f"[INFO] Precomputed {len(self.score_cache)} resume/JD pairs.")

    st.info(f"[INFO] Crosschecker model trained and scored. Model saved at {self.model_path}")

  def compute_relevance(self, resume_text, jd_text):
    """
    Takes a resume text and JD text, computes relevance score using BERT-based crosschecker.
    Returns a float between 0 and 1 (probability the texts 'match').
    """
    self.model.eval()
    encodings = self.tokenizer([resume_text], [jd_text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
      outputs = self.model(
        encodings["input_ids"].to(self.device),
        attention_mask=encodings["attention_mask"].to(self.device)
      )
      probs = torch.softmax(outputs.logits, -1)
      relevance = probs[0,1].item()  # Probability of label==1 ("relevant")
    return float(relevance)

  def get_scores(self, applicant_name, resume_text, jd_text):
    """
    API for the Streamlit UI. Looks up completeness/truthiness/relevance scores by resume/JD.
    Assumes resume_text and jd_text correspond to user selections.
    """
    # Try to get precomputed score first
    key = (applicant_name, jd_text[:50])
    if key in self.score_cache:
      return self.score_cache[key]

    # Fallback: compute completeness and truthiness naively
    # Try to find the resume in any of the training files
    resume_df = None
    for resume_file in self.resume_files:
      if os.path.exists(resume_file):
        temp_df = pd.read_csv(resume_file)
        if resume_df is None:
          resume_df = temp_df
        else:
          resume_df = pd.concat([resume_df, temp_df], ignore_index=True)
    if resume_df is None:
      resume_df = pd.DataFrame(columns=['text', 'completeness', 'truthiness'])

    # Find the resume by name, else by closest match
    found_row = None
    if "name" in resume_df.columns and applicant_name:
      matches = resume_df[resume_df["name"].str.match(applicant_name, case=False, na=False)]
      if not matches.empty:
        found_row = matches.iloc[0]
    if found_row is None:
      # fallback: use first row whose text matches
      matches = resume_df[resume_df["text"] == resume_text]
      if not matches.empty:
        found_row = matches.iloc[0]
    if found_row is None:
      found_row = resume_df.iloc[0]  # last fallback

    completeness = float(found_row.get("completeness", 0))
    truthiness = float(found_row.get("truthiness", 0))

    relevance = self.compute_relevance(resume_text, jd_text)
    # Return all as 0-100 scale for UI
    return {
      "completeness": completeness * 100 if completeness < 1.01 else completeness,
      "truthiness": truthiness * 100 if truthiness < 1.01 else truthiness,
      "relevance": relevance * 100
    }

# For integration: supports Streamlit app and compatibility
def create_initial_dataset():
  """
  NO-OP, since the resume_final.csv is expected to be created/updated upstream by completeness & truthiness models.
  """
  pass

if __name__ == "__main__":
  CrossCheckerModel().run()

prev = '''import os
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
      st.warning(f"Warning: CSV load failed: {e}. Creating empty dataframe.")
      self.df = pd.DataFrame(columns=['name', 'completeness', 'truthiness'])

    self.tokenizer = AutoTokenizer.from_pretrained(emb_model)
    # self.model = AutoModel.from_pretrained(emb_model).to("cpu")
    self.model = AutoModel.from_pretrained(emb_model, torch_dtype=torch.float32, device_map=None).to("cpu")

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
'''
