#%%
import requests
from requests.auth import HTTPBasicAuth
import os
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import getpass
 
#%%
def download_matching_files(base_url, username, password, pattern="t2m_elvcorr_2001.*.nc", dest_path="downloads"):
    """
    Download files matching the pattern from an HTTPS server with authentication
   
    Parameters:
    -----------
    base_url : str
        URL of the server directory
    username : str
        Username for authentication
    password : str
        Password for authentication
    pattern : str
        Regex pattern to match filenames
    dest_path : str
        Destination directory path for downloaded files
    """
    # Create a session with authentication
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
   
    # Create destination directory
    os.makedirs(dest_path, exist_ok=True)
   
    print(f"Connecting to {base_url}...")
   
    try:
        # Try to get directory listing
        response = session.get(base_url)
        response.raise_for_status()
       
        # Parse HTML to find links
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [link.get('href') for link in soup.find_all('a')]
 
        # split links to get only the file names
        links = [link.split('/')[-1] for link in links]
       
        # Filter links matching our pattern
        matching_files = [link for link in links if link and re.match(pattern, link)]
       
        if not matching_files:
            print("No matching files found in directory listing.")
            return
           
        print(f"Found {len(matching_files)} matching files.")
       
        # Download each matching file
        for file_name in matching_files:
            file_url = urljoin(base_url, file_name)
            local_path = os.path.join(dest_path, file_name)
           
            print(f"Downloading {file_name}...")
            try:
                response = session.get(file_url, stream=True)
                response.raise_for_status()
               
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
               
                print(f"Successfully downloaded {file_name}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_name}: {e}")
           
    except requests.exceptions.RequestException as e:
        print(f"Error accessing server: {e}")
 
if __name__ == "__main__":
    server_url = input("Enter server URL (e.g., https://example.com/data/): ")
    username = input("Enter username: ")
    password = getpass.getpass("Enter password: ")
    dest_path = input("Enter destination directory path [downloads]: ") or "downloads"
    pattern = input("Enter file pattern e.g., [t2m_elvcorr_2000.*.nc]: ") or r"t2m_elvcorr_2000.*.nc"
    download_matching_files(server_url, username, password, pattern, dest_path)