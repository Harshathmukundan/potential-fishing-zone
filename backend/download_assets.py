import os
import shutil
import glob
import zipfile
import gdown

def download_and_extract_smart(file_id, output_zip, target_dir, extensions):
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading {output_zip} from Google Drive...")
    gdown.download(url, output_zip, quiet=False)
    
    if not os.path.exists(output_zip):
        print(f"Failed to download {output_zip}")
        return

    temp_dir = output_zip + "_temp_extract"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Extracting {output_zip}...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    print(f"Moving files to {target_dir}...")
    found_files = False
    for ext in extensions:
        for filepath in glob.glob(os.path.join(temp_dir, '**', f'*{ext}'), recursive=True):
            filename = os.path.basename(filepath)
            dest = os.path.join(target_dir, filename)
            # Only move if the file doesn't already exist or if we want to overwrite
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(filepath, dest)
            print(f" -> Moved {filename}")
            found_files = True
            
    if not found_files:
        print("⚠️ Warning: No files matching the requested extensions were found in the zip!")

    # Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    os.remove(output_zip)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Download saved_models
    saved_models_id = '1Iqe-Nbqil8H4sHoMi-rIsbgaBc0-6Ds2'
    saved_models_zip = os.path.join(base_dir, 'saved_models_temp.zip')
    saved_models_dir = os.path.join(base_dir, 'saved_models')
    print("=== Processing Models ===")
    download_and_extract_smart(saved_models_id, saved_models_zip, saved_models_dir, ['.keras', '.h5', '.json'])
    
    # 2. Download DATA
    data_id = '10xN-Va1UygdKol3aoF35iFUEJUvikAyI'
    data_zip = os.path.join(base_dir, 'DATA_temp.zip')
    data_dir = os.path.join(base_dir, 'uploads', 'DATA')
    print("\n=== Processing Datasets ===")
    download_and_extract_smart(data_id, data_zip, data_dir, ['.nc'])
    
    print("\n✅ All assets downloaded and extracted successfully!")
