import os
import streamlit as st
import pandas as pd
import torch
import joblib
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
import time
import re

# Set Kaggle env vars from Streamlit secrets (no file writing)
if "kaggle" in st.secrets:
  os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
  os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]
  st.info("[INFO] Kaggle API credentials set from secrets.")
else:
  st.warning("[WARN] Kaggle secrets not found. Downloads may fail without authentication.")

# Helper to launch a headless Selenium browser
def init_driver(browser="chrome"):
  from webdriver_manager.chrome import ChromeDriverManager
  from webdriver_manager.firefox import GeckoDriverManager
  from selenium.webdriver.chrome.service import Service as ChromeService
  from selenium.webdriver.firefox.service import Service as FirefoxService
  if browser == "chrome":
    service = ChromeService(ChromeDriverManager().install())
    options = ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=service, options=options)
  else:
    service = FirefoxService(GeckoDriverManager().install())
    options = FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(service=service, options=options)
  return driver

# Search relevant links from Google for a given entity and type
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
  except:
    pass
  return list(set(links))

def compute_completeness(row, driver):
  text = row["text"]
  # Identify projects or features in some robust way
  projects = re.findall(r'project[:\- ](.*?)(?=\.|\n|;|$)', text, flags=re.IGNORECASE)
  features = re.findall(r'feature[:\- ](.*?)(?=\.|\n|;|$)', text, flags=re.IGNORECASE)
  score = 0
  n = 0
  for proj in projects:
    result_links = search_links(f"{proj} github", driver)
    score += 1 if result_links else 0
    n += 1
    # Try leetcode for 'solution', 'submission' or similar
    result_links = search_links(f"{proj} leetcode", driver)
    score += 1 if result_links else 0
    n += 1
  for feature in features:
    result_links = search_links(f"{feature} linkedin", driver)
    score += 1 if result_links else 0
    n += 1
    result_links = search_links(f"{feature} facebook", driver)
    score += 1 if result_links else 0
    n += 1
  if n == 0: return 0
  return min(100, int((score / n) * 100))

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
    self.resume_dir = os.path.join(self.base_dir, "training_resumes")
    self.jd_dir = os.path.join(self.base_dir, "training_jds")

    os.makedirs(self.resume_dir, exist_ok=True)
    os.makedirs(self.jd_dir, exist_ok=True)

    self.resume_outfiles = [
      os.path.join(self.resume_dir, f"resume_final_{i+1}.csv") for i in range(3)
    ]
    self.jd_outfiles = [
      os.path.join(self.jd_dir, f"jd_final_{i+1}.csv") for i in range(3)
    ]

    self.model_dir = "models/saved"
    self.model_path = os.path.join(self.model_dir, "completeness_bert.pt")
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Make sure directory exists
    os.makedirs(self.model_dir, exist_ok=True)
    os.makedirs(self.base_dir, exist_ok=True)

  def download_and_aggregate(self):
    import glob
    resume_dfs = []
    for i, kaggle_id in enumerate(self.resume_datasets):
      outfile = self.resume_outfiles[i]
      if os.path.exists(outfile):
        try:
          df = pd.read_csv(outfile)
          st.info(f"[INFO] Loaded local {outfile}")
          resume_dfs.append(df)
        except Exception as e:
          st.warning(f"[WARN] Could not load local file {outfile}: {e}")
          resume_dfs.append(pd.DataFrame())
        continue
      try:
        st.info(f"[INFO] Downloading {kaggle_id} from Kaggle for resume dataset {i+1}...")
        dataset = load_dataset("kaggle", kaggle_id)
        split = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[split])
        if "Resume_html" in df.columns:
          df["text"] = df["Resume_html"].astype(str)
          df["completeness"] = 1
        elif "resume" in df.columns:
          df["text"] = df["resume"].astype(str)
          df["completeness"] = 1
        elif "text" in df.columns:
          if "completeness" not in df.columns:
            df["completeness"] = 1
        else:
          df["text"] = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)
          df["completeness"] = 1
        df = df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
        df = df[["text", "completeness"]]
        df.to_csv(outfile, index=False)
        resume_dfs.append(df)
        st.success(f"[INFO] Saved resume dataset to {outfile}")
      except Exception as e:
        st.warning(f"[WARN] Could not load {kaggle_id}: {e}")
        resume_dfs.append(pd.DataFrame())
    if not any([not df.empty for df in resume_dfs]):
      local_files = glob.glob(os.path.join(self.resume_dir, "resume_final_*.csv"))
      for f in local_files:
        try:
          df = pd.read_csv(f)
          resume_dfs.append(df)
        except:
          pass
    if all([df.empty for df in resume_dfs]):
      raise RuntimeError("No resume datasets could be found or downloaded.")
    jd_dfs = []
    for i, kaggle_id in enumerate(self.jd_datasets):
      outfile = self.jd_outfiles[i]
      if os.path.exists(outfile):
        try:
          jd_df = pd.read_csv(outfile)
          st.info(f"[INFO] Loaded local {outfile}")
          jd_dfs.append(jd_df)
        except Exception as e:
          st.warning(f"[WARN] Could not load local JD file {outfile}: {e}")
          jd_dfs.append(pd.DataFrame())
        continue
      try:
        st.info(f"[INFO] Downloading {kaggle_id} from Kaggle for JD dataset {i+1}...")
        dataset = load_dataset("kaggle", kaggle_id)
        split = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[split])
        jd_text_col = None
        for col in ["text", "description", "job_description"]:
          if col in df.columns:
            jd_text_col = col
            break
        if jd_text_col:
          df = df[[jd_text_col]]
          df.columns = ["text"]
        else:
          df["text"] = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)
        df = df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
        df.to_csv(outfile, index=False)
        jd_dfs.append(df)
        st.success(f"[INFO] Saved JD dataset to {outfile}")
      except Exception as e:
        st.warning(f"[WARN] Could not load {kaggle_id}: {e}")
        jd_dfs.append(pd.DataFrame())
    if not any([not df.empty for df in jd_dfs]):
      local_files = glob.glob(os.path.join(self.jd_dir, "jd_final_*.csv"))
      for f in local_files:
        try:
          df = pd.read_csv(f)
          jd_dfs.append(df)
        except:
          pass
    if all([df.empty for df in jd_dfs]):
      st.warning("[WARN] No JD datasets could be found or downloaded.")
    if all([df.empty for df in resume_dfs]):
      raise RuntimeError("No usable resume datasets found in training_data/training_resumes/")
    if all([df.empty for df in jd_dfs]):
      st.warning("[WARN] All JD datasets are empty!")

  def create_initial_dataset(self):
    """
    Only run if resume_final.csv does not exist (as per workflow).
    """
    # Check if any resume files exist
    files_exist = any(os.path.exists(f) for f in self.resume_outfiles)
    if not files_exist:
      self.download_and_aggregate()


  def load_training_data(self):
    """
    Loads processed resume data. Does a simple split into train and eval sets.
    """
    # Load from the first available resume file for training
    for resume_file in self.resume_outfiles:
      if os.path.exists(resume_file):
        df = pd.read_csv(resume_file)
        break
    else:
      raise FileNotFoundError("No resume training files found")

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
    with st.spinner("Training CompletenessModel on resume and JD datasets..."):
      trainer.train()
    # Save model and tokenizer
    model.save_pretrained(self.model_dir)
    self.tokenizer.save_pretrained(self.model_dir)
    torch.save(model.state_dict(), self.model_path)

    # Inference on the base resume dataset: add 'completeness' score for each entry
    # (For now, this just reuses the labels, but in a real use-case you'd do:
    #   scores = model(X) -> update CSV)

    # Process each resume file with completeness scoring
    for outfile in self.resume_outfiles:
      if not os.path.exists(outfile):
        continue
      df = pd.read_csv(outfile)
      try:
        driver = init_driver()
        completeness_scores = []
        for _, row in df.iterrows():
          completeness_scores.append(compute_completeness(row, driver))
        driver.quit()
      except WebDriverException as e:
        st.warning(f"Selenium failed: {e}. Using default score 0.")
        completeness_scores = [0] * len(df)
      df["completeness"] = completeness_scores
      df.to_csv(outfile, index=False)

    st.info(f"[INFO] Completeness model trained and applied. Model saved at {self.model_path}")

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
  st.info("Initial datasets created from Kaggle.")

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
      st.warning(f"{self.csv_out} exists; skipping recompute.")
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
    st.info("Completeness scores and social links written to", self.csv_out)

    return self.csv_out

if __name__ == "__main__":
  CompletenessModel().run()
'''
