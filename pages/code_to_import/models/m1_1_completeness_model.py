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
BROWSER = "chrome"  # or "firefox"

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
    if BROWSER == "chrome":  # Fallback to Firefox if Chrome fails
      try:
        ffopts = FFOpts()
        ffopts.add_argument("--headless")
        driver = webdriver.Firefox(options=ffopts)
        return driver
      except:
        raise RuntimeError("Both Chrome and Firefox drivers failed")
    else:
      raise RuntimeError("Driver initialization failed")

def extract_name_ner(text):
  ner_pipe = pipeline("ner", model=NER_MODEL, grouped_entities=True)
  ent = ner_pipe(text[:400])
  for e in ent:
    if e['entity_group'] == "PER":
      return e['word']
  return None

def search_social_links(name, orgs=None, driver=None):
  """
  Given a name, auto-search GitHub/LinkedIn/Facebook and return urls if found. Optionally search organization names too.
  """
  if not name: return {'github': '', 'linkedin': '', 'facebook': ''}
  links = {}
  queries = {
    'github': f'"{name}" github',
    'linkedin': f'"{name}" linkedin',
    'facebook': f'"{name}" facebook'
  }
  results = {}
  for site, q in queries.items():
    for attempt in range(2):  # Retry once
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
  """Search for statements like 'completed X project' or project deployment artifacts."""
  if not project or len(project) < 10: return 0.0
  query = f"{project} deployed OR completed OR github OR site OR final"
  driver.get(f"https://google.com/search?q={query.replace(' ','+')}")
  time.sleep(1.0)
  # Look for deployment pages or success confirmations
  page = driver.page_source.lower()
  return 1.0 if ("deployed" in page or "final" in page or "completed" in page) else 0.5

class CompletenessModel:
  REQUIRED_SECTIONS = ['education', 'skills', 'experience', 'projects']

  def __init__(self, csv_in="resume_data.csv", csv_out="resume_augmented.csv"):
    self.csv_in = csv_in
    self.csv_out = csv_out

  def run(self):
    if os.path.exists(self.csv_out):
      print(f"{self.csv_out} exists; skipping recompute.")
      return self.csv_out
    df = pd.read_csv(self.csv_in)
    driver = init_driver()
    tqdm.pandas()
    # For each resume: auto-find missing social links, check project status
    def process_row(row):
      txt = str(row.get("text", ""))[:2500]
      name = extract_name_ner(txt)
      links = str(row.get("links", ""))
      # Autofind links if missing
      if not ("github" in links or "linkedin" in links):
        found = search_social_links(name, driver=driver)
        for s in ["github", "linkedin", "facebook"]:
          if found[s] and found[s] not in links:
            links += "," + found[s]
      # Completeness: how many of REQUIRED_SECTIONS and if project(s) are "really" completed
      completeness = sum(f in txt.lower() for f in self.REQUIRED_SECTIONS) / len(self.REQUIRED_SECTIONS)
      projects = row.get("projects", "")
      if projects:
        completeness += 0.2 * check_project_status_via_web(projects, driver)
      return pd.Series([links.strip(","), min(100, round(completeness * 100, 2))])
    df[["links", "completeness"]] = df.progress_apply(process_row, axis=1)
    df.to_csv(self.csv_out, index=False)
    driver.quit()
    print("Completeness scores and social links written to", self.csv_out)
    return self.csv_out

if __name__ == "__main__":
  CompletenessModel().run()
