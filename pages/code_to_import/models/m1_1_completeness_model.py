import os
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
