import os, sys, re, datetime, csv, zipfile, shutil

def delete_zip_files():
  curr_dir = os.getcwd()
  for folders, _, files in os.walk(curr_dir):
    for file in files:
      if re.search(r'^zippy_', file):
        os.remove(os.path.join(folders, file))
  # print("Zip files deleted from repo.")

def find_downloads_folder():
  downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
  if os.path.isdir(downloads_path):
    pass
  else:
    # make our own path
    os.makedirs(os.path.join(os.path.expanduser("~"), "Downloads"), exist_ok=True)
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")

  return downloads_path

def download_all_files_flat_to_downloads():
  ts = datetime.datetime.today()

  zipper_file_name = f"zippy_repo_backup_{ts.day}_{ts.month}_{ts.year}_{ts.hour}_{ts.minute}_{ts.second}.zip"
  curr_dir = os.getcwd()
  with zipfile.ZipFile(zipper_file_name, 'w') as zippy:
    for folders, sub_f, files in os.walk(curr_dir):
      # Skip __pycache__ directories
      sub_f[:] = [d for d in sub_f if d not in ("__pycache__", ".git", "JDs", "resume_and_supporting_docs")]

      for file in files:
        # Skip README files and zip files starting with zippy_
        if file.lower() in ["readme.md", "readme.txt", "readme.rst", "__init__.py", "download_all_as_zip.py"] or re.match(r'^zippy_', file):
          continue
        full_path = os.path.join(folders, file)
        # Write file to zip with arcname as basename (flat structure)
        zippy.write(full_path, arcname=os.path.basename(full_path), compress_type=zipfile.ZIP_DEFLATED)

  # Step 2: Extract the zip file to a subfolder in the Downloads folder
  downloads_path = find_downloads_folder()

  # Create a new subfolder inside Downloads with the same timestamp in the folder name
  folder_name = f"repo_backup_{ts.day}_{ts.month}_{ts.year}_{ts.hour}_{ts.minute}_{ts.second}"
  target_folder = os.path.join(downloads_path, folder_name)
  os.makedirs(target_folder, exist_ok=True)

  with zipfile.ZipFile(zipper_file_name, 'r') as unzippy:
    unzippy.extractall(path=target_folder)

  print(f"\nDownload of file: {zipper_file_name} complete! Check your downloads folder :)")

  # Step 3: Delete the zip file from the repo
  delete_zip_files()

if __name__ == "__main__":
  download_all_files_flat_to_downloads()
