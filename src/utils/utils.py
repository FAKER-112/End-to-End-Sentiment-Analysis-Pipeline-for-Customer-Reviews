import requests
import os

def download_file(url, dest_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created directory: {dest_folder}")

    # Get the file name from the URL
    filename = url.split('/')[-1]
    file_path = os.path.join(dest_folder, filename)
    return file_path