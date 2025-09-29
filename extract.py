import zipfile

zip_path = "celeba-dataset.zip"
out_dir = "data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(out_dir)

print("✅ Extraction complete!")
