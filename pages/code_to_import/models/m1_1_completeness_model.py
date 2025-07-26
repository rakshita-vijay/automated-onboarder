import os
import pandas as pd
import torch
import joblib
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

class CompletenessModel:
  """
  Model to assess 'completeness' of projects/claims in resumes.
  - Downloads 3 resume and 3 JD datasets from Kaggle.
  - Aggregates and cleans these datasets as training data.
  - Fine-tunes BERT for binary/multi-class completeness classification.
  - Adds a 'completeness' column to 'resume_final.csv'.
  - Saves trained model to disk for later inference.
  """

  def __init__(self):
    self.resume_datasets = [
      "snehaanbhawal/resume-dataset",
      "amananandrai/resume-entities-for-ner",
      "dyneins/resume-dataset"
    ]
    self.jd_datasets = [
      "hiringtruck/job-description-dataset",
      "promptcloud/indeed-job-posting-dataset",
      "navoneel/industry-relevant-resume-phrases"
    ]
    self.base_dir = "training_data"
    self.resume_outfile = "resume_final.csv"
    self.jd_outfile = "jd_final.csv"
    self.model_dir = "models/saved"
    self.model_path = os.path.join(self.model_dir, "completeness_bert.pt")
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Make sure directory exists
    os.makedirs(self.model_dir, exist_ok=True)
    os.makedirs(self.base_dir, exist_ok=True)

  def download_and_aggregate(self):
    """
    Download resume and JD datasets from Kaggle via HuggingFace Datasets.
    All files are concatenated into single DataFrames and saved.
    """
    # Resume data aggregation
    resume_dfs = []
    for kaggle_id in self.resume_datasets:
      try:
        dataset = load_dataset("kaggle", kaggle_id)
        # Take only first available split for simplicity
        split = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[split])
        resume_dfs.append(df)
      except Exception as e:
        print(f"[WARN] Could not load {kaggle_id}: {e}")
    if not resume_dfs:
      raise RuntimeError("No resume datasets could be loaded.")

    # Try to extract a sensible column for training
    for df in resume_dfs:
      # Add 'text' and 'label' (for completeness) columns where possible
      if "Resume_html" in df.columns:
        df["text"] = df["Resume_html"].astype(str)
        df["completeness"] = 1  # Assume all entries 'complete'; will be refined
      elif "resume" in df.columns:
        df["text"] = df["resume"].astype(str)
        df["completeness"] = 1
      elif "text" in df.columns:
        if "completeness" not in df.columns:
          df["completeness"] = 1
      else:
        # Fallback: concatenate all columns as text
        df["text"] = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)
        df["completeness"] = 1
    resume_df = pd.concat(resume_dfs, ignore_index=True)
    # Remove duplicates, drop NaN
    resume_df = resume_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
    resume_df = resume_df[["text", "completeness"]]

    # JD aggregation
    jd_dfs = []
    for kaggle_id in self.jd_datasets:
      try:
        dataset = load_dataset("kaggle", kaggle_id)
        split = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[split])
        jd_dfs.append(df)
      except Exception as e:
        print(f"[WARN] Could not load {kaggle_id}: {e}")

    if jd_dfs:
      jd_df = pd.concat(jd_dfs, ignore_index=True)
      # Prefer a column called "text", "description", or fallback
      jd_text_col = None
      for col in ["text", "description", "job_description"]:
        if col in jd_df.columns:
          jd_text_col = col
          break
      if jd_text_col:
        jd_df = jd_df[[jd_text_col]]
        jd_df.columns = ["text"]
      else:
        jd_df["text"] = jd_df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)
      jd_df = jd_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
      jd_df.to_csv(os.path.join(self.base_dir, self.jd_outfile), index=False)  # Save as reference
    else:
      print("[WARN] No JD datasets loaded, skipping.")

    # Save resume (base) dataset
    resume_df.to_csv(os.path.join(self.base_dir, self.resume_outfile), index=False)

  def create_initial_dataset(self):
    """
    Only run if resume_final.csv does not exist (as per workflow).
    """
    outfile = os.path.join(self.base_dir, self.resume_outfile)
    if not os.path.exists(outfile):
      self.download_and_aggregate()

  def load_training_data(self):
    """
    Loads processed resume data. Does a simple split into train and eval sets.
    """
    df = pd.read_csv(os.path.join(self.base_dir, self.resume_outfile))
    # If 'completeness' is missing or not binary, default to 1 for all (labeling needed later)
    if "completeness" not in df.columns:
      df["completeness"] = 1
    # Truncate or clean as required for demonstration
    df = df[df["text"].str.len() > 50]  # Ignore trivially short texts
    return train_test_split(df["text"], df["completeness"], test_size=0.2, random_state=42)

  def tokenize_function(self, texts):
    """
    Helper for tokenizer batch processing.
    """
    return self.tokenizer(
      list(texts), padding=True, truncation=True, max_length=256, return_tensors='pt'
    )

  def run(self):
    """
    Main entrypoint called at app startup.
    - Downloads and prepares data if not present.
    - Trains (fine-tunes) a BERT classifier, saves model for later use.
    - Adds/updates the 'completeness' column in resume_final.csv.
    """
    self.create_initial_dataset()

    # Prepare data for fine-tuning the BERT model
    X_train, X_eval, y_train, y_eval = self.load_training_data()
    train_encodings = self.tokenize_function(X_train)
    eval_encodings = self.tokenize_function(X_eval)
    train_labels = torch.tensor(list(y_train))
    eval_labels = torch.tensor(list(y_eval))

    class ResumeDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
      def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
      def __len__(self):
        return len(self.labels)

    train_dataset = ResumeDataset(train_encodings, train_labels)
    eval_dataset = ResumeDataset(eval_encodings, eval_labels)

    # Setup model
    model = BertForSequenceClassification.from_pretrained(
      "bert-base-uncased", num_labels=2  # 2: complete/incomplete (expandable)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training arguments
    training_args = TrainingArguments(
      output_dir=self.model_dir,
      do_train=True,
      do_eval=True,
      num_train_epochs=1,              # Quick demo/train at startup
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

    # Train BERT model on completeness task
    print("Training CompletenessModel on resume and JD datasets...")
    trainer.train()
    # Save model and tokenizer
    model.save_pretrained(self.model_dir)
    self.tokenizer.save_pretrained(self.model_dir)
    torch.save(model.state_dict(), self.model_path)

    # Inference on the base resume dataset: add 'completeness' score for each entry
    # (For now, this just reuses the labels, but in a real use-case you'd do:
    #   scores = model(X) -> update CSV)
    outfile = os.path.join(self.base_dir, self.resume_outfile)
    df = pd.read_csv(outfile)
    # Re-predict completeness with trained model for all entries
    encodings = self.tokenize_function(df["text"])
    with torch.no_grad():
      model.eval()
      outputs = model(
        encodings["input_ids"].to(device),
        attention_mask=encodings["attention_mask"].to(device)
      )
      preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
    # Set new completeness column
    df["completeness"] = preds
    df.to_csv(outfile, index=False)
    print(f"[INFO] Completeness model trained and applied. Model saved at {self.model_path}")

# Support function for application_evaluator.py
def create_initial_dataset():
  """
  Helper to ensure dataset exists for startup call (matches workflow and code in application_evaluator.py).
  """
  CompletenessModel().create_initial_dataset()

# For manual testing (not required in app, but left for standalone runs)
if __name__ == "__main__":
  CompletenessModel().run()

prev = '''import os
import pandas as pd
import re
import time
from tqdm.auto import tqdm
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.options import Options as FFOpts
from selenium.common.exceptions import WebDriverException

NER_MODEL = "Babelscape/wikineural-multilingual-ner"

TRAIN_RESUMES_DIR = "training_data/training_resumes"
os.makedirs(TRAIN_RESUMES_DIR, exist_ok=True)

# Change all csv_in/csv_out to use TRAIN_RESUMES_DIR, e.g.:
self.csv_in = os.path.join(TRAIN_RESUMES_DIR, "resume_data.csv")
self.csv_out = os.path.join(TRAIN_RESUMES_DIR, "resume_augmented.csv")

BROWSER = "chrome"

def download_kaggle_datasets():
  import kaggle
  resume_datasets = ["snehaanbhawal/resume-dataset", "amananandrai/resume-entities-for-ner", "dyneins/resume-dataset"]
  jd_datasets = ["hiringtruck/job-description-dataset", "promptcloud/indeed-job-posting-dataset", "navoneel/industry-relevant-resume-phrases"]
  all_data = {"resumes": [], "jds": []}
  # Serially download resumes
  for ds in resume_datasets:
    ddir = os.path.join(TRAIN_RESUMES_DIR, ds.split("/")[-1])
    kaggle.api.dataset_download_files(ds, path=ddir, unzip=True)
    for root, _, files in os.walk(ddir):
      for f in files:
        if f.endswith(('.txt', '.csv', '.pdf', '.docx')):
          fpath = os.path.join(root, f)
          if f.endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8') as fin:
              all_data["resumes"].append(fin.read())

          elif f.endswith('.pdf'):
            all_data["resumes"].append(extract_pdf(fpath))

          elif f.endswith('.docx'):
            all_data["resumes"].append(extract_docx(fpath))

          elif f.endswith('.csv'):
            try:
              df_csv = pd.read_csv(fpath)
              text_cols = [c for c in df_csv.columns if any(k in c.lower() for k in ['resume', 'text', 'description', 'content'])]
              if text_cols:
                for col in text_cols:
                  all_data["resumes"].extend(df_csv[col].dropna().astype(str).tolist())
              else:
                all_data["resumes"].extend(df_csv.astype(str).agg(' '.join, axis=1).tolist())
            except:
              pass

  # Serially download JDs
  for ds in jd_datasets:
    ddir = os.path.join(TRAIN_JDS_DIR, ds.split("/")[-1])
    kaggle.api.dataset_download_files(ds, path=ddir, unzip=True)
    for root, _, files in os.walk(ddir):
      for f in files:
        if f.endswith(('.txt', '.csv', '.pdf', '.docx')):
          fpath = os.path.join(root, f)
          if f.endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8') as fin:
              all_data["jds"].append(fin.read())

          elif f.endswith('.pdf'):
            all_data["jds"].append(extract_pdf(fpath))

          elif f.endswith('.docx'):
            all_data["jds"].append(extract_docx(fpath))

          elif f.endswith('.csv'):
            try:
              df_csv = pd.read_csv(fpath)
              text_cols = [c for c in df_csv.columns if any(k in c.lower() for k in ['job_description', 'text', 'description', 'content'])]
              if text_cols:
                for col in text_cols:
                  all_data["jds"].extend(df_csv[col].dropna().astype(str).tolist())
              else:
                all_data["jds"].extend(df_csv.astype(str).agg(' '.join, axis=1).tolist())
            except:
              pass

  return all_data

def create_initial_dataset():
  if os.path.exists("resume_data.csv"):
    return

  data = download_kaggle_datasets()
  resume_df = pd.DataFrame({"text": data["resumes"], "projects": "", "links": "", "name": ""})  # Dummy names; add NER if needed
  resume_df.to_csv(os.path.join(TRAIN_RESUMES_DIR, "resume_data.csv"), index=False)
  jd_df = pd.DataFrame({"text": data["jds"]})
  jd_df.to_csv(os.path.join(TRAIN_JDS_DIR, "jd_data.csv"), index=False)
  print("Initial datasets created from Kaggle.")


def init_driver():
  options = Options()
  options.add_argument("--headless")
  options.add_argument("--no-sandbox")
  try:
    if BROWSER == "chrome":
      driver = webdriver.Chrome(options=options)
    else:
      ffopts = FFOpts()
      ffopts.add_argument("--headless")
      driver = webdriver.Firefox(options=ffopts)
    return driver
  except WebDriverException as e:
    if BROWSER == "chrome":
      ffopts = FFOpts()
      ffopts.add_argument("--headless")
      driver = webdriver.Firefox(options=ffopts)
      return driver
    raise RuntimeError("Driver initialization failed")

def extract_name_ner(text):
  ner_pipe = pipeline("ner", model=NER_MODEL, grouped_entities=True)
  ent = ner_pipe(text[:400])
  for e in ent:
    if e['entity_group'] == "PER":
      return e['word']
  return None

def search_social_links(name, orgs=None, driver=None):
  if not name: return {'github': '', 'linkedin': '', 'facebook': ''}
  queries = {
    'github': f'"{name}" github',
    'linkedin': f'"{name}" linkedin',
    'facebook': f'"{name}" facebook'
  }
  results = {}
  for site, q in queries.items():
    for attempt in range(3):
      try:
        driver.get(f"https://google.com/search?q={q.replace(' ','+')}")
        time.sleep(1.5)
        res = driver.find_elements("css selector", "a")
        url = next((elem.get_attribute("href")
                    for elem in res if site in elem.get_attribute("href")), '')
        results[site] = url
        break
      except:
        time.sleep(1)
  return results

def check_project_status_via_web(project, driver):
  if not project or len(project) < 10: return 0.0
  query = f"{project} deployed OR completed OR github OR site OR final"
  driver.get(f"https://google.com/search?q={query.replace(' ','+')}")
  time.sleep(1.0)
  page = driver.page_source.lower()
  return 1.0 if ("deployed" in page or "final" in page or "completed" in page) else 0.5

class CompletenessModel:
  REQUIRED_SECTIONS = ['education', 'skills', 'experience', 'projects']

  def __init__(self, csv_in="resume_data.csv", csv_out="resume_augmented.csv"):
    self.csv_in = csv_in
    self.csv_out = csv_out

  def run(self):
    create_initial_dataset()

    if os.path.exists(self.csv_out):
      print(f"{self.csv_out} exists; skipping recompute.")
      return self.csv_out

    df = pd.read_csv(self.csv_in)
    driver = init_driver()
    tqdm.pandas()

    def process_row(row):
      txt = str(row.get("text", ""))[:2500]

      # Extract projects section for verification
      import re
      proj_match = re.search(r"project[s]*[:\-](.*?)(?=(\n\n|\Z))", txt, re.DOTALL | re.IGNORECASE)
      projects = proj_match.group(1).strip() if proj_match else ""

      name = extract_name_ner(txt) or row.get("name", "")
      links = str(row.get("links", ""))
      if not ("github" in links or "linkedin" in links):
        found = search_social_links(name, driver=driver)
        for s in ["github", "linkedin", "facebook"]:
          if found[s] and found[s] not in links:
            links += "," + found[s]

      completeness = sum(f in txt.lower() for f in self.REQUIRED_SECTIONS) / len(self.REQUIRED_SECTIONS)
      if projects:
        completeness += 0.2 * check_project_status_via_web(projects, driver)
      completeness = min(100, round(completeness * 100, 2))
      return pd.Series([name, links.strip(","), projects, completeness])

    df[["name", "links", "projects", "completeness"]] = df.progress_apply(process_row, axis=1)
    df.to_csv(self.csv_out, index=False)
    driver.quit()
    print("Completeness scores and social links written to", self.csv_out)

    return self.csv_out

if __name__ == "__main__":
  CompletenessModel().run()
'''
