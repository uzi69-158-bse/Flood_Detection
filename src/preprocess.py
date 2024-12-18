import requests
import os
import zipfile

API_KEY = "YOUR_NVIDIA_API_KEY"
BASE_URL = "https://api.nvidia.com/cloud-storage"

dataset_id = "flood_dataset"
output_dir = "data"

def download_data():
    url = f"{BASE_URL}/datasets/{dataset_id}/download"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        zip_path = os.path.join(output_dir, f"{dataset_id}.zip")
        os.makedirs(output_dir, exist_ok=True)

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print("Failed to download data:", response.status_code)

if __name__ == "__main__":
    download_data()
