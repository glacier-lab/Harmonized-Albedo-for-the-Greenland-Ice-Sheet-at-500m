
'''
Download files matching a filename pattern from an HTTPS server (optionally authenticated).

This function connects to a web directory or THREDDS catalog page, discovers candidate
file URLs, filters them by a regular-expression filename pattern, and downloads each
matching file into a local directory.

Behavior and special handling:
- Parses the provided base_url HTML and inspects <a href="..."> links.
- Recognizes two link types:
    1. THREDDS catalog links that include a "dataset=..." query parameter. These are
         converted to a THREDDS fileServer URL by replacing the catalog path with
         "<scheme>://<netloc>/thredds/fileServer/<dataset>".
    2. Direct .nc file links (absolute or relative). Relative links are resolved
         against base_url with urllib.parse.urljoin.
- Filters discovered files by matching the basename against the provided regular
    expression pattern (Python re.match semantics).
- Downloads each matching file in streaming mode to avoid loading entire files into memory.
- Creates dest_path if it does not exist.
- Prints progress and error messages. Network and HTTP errors are caught and reported;
    the function does not raise these exceptions.

Parameters
----------
        URL of the server directory or THREDDS catalog page (e.g.
        "https://example.com/data/" or a THREDDS catalog HTML page).
username : str or None, optional
        Username for HTTP Basic authentication. If None or empty, authentication is not used.
password : str or None, optional
        Password for HTTP Basic authentication. If None or empty, authentication is not used.
pattern : str, optional
        Regular-expression pattern used to match filenames (applied to the basename of each
        discovered file). Example: r"SICEv3.0_.*\.nc". Default: r"*.nc" (note: this default is a
        glob-like pattern but should be provided as a valid regular expression for reliable results).
dest_path : str, optional
        Local directory path where matching files will be saved. Default: "downloads".

Returns
-------
None
        The function performs downloads as a side effect and reports status via printing.
        It does not return a value.

Notes
-----
**It was developed and tested on Ubuntu 20.04.**
- This routine uses requests.Session and will attach HTTPBasicAuth to the session only
    if both username and password are provided.
- The THREDDS conversion assumes the server exposes a "/thredds/fileServer/" path at the
    same host as the provided catalog URL.
- Filenames are matched using re.match against the file basename; if you want to test
    for a pattern anywhere in the filename, adjust the pattern accordingly (e.g. use ".*pattern.*").
- For large files, downloads are streamed in chunks of 8192 bytes to disk.

Example
-------
download_matching_files(
        "https://thredds.example.org/thredds/catalog/collection/catalog.html",
        username=None,
        password=None,
        pattern=r"SICEv3.0_.*\.nc",
        dest_path="/path/to/save"
)

Author: Shunan Feng (shunan.feng@envs.au.dk)
'''
#%%
import requests
from requests.auth import HTTPBasicAuth
import os
import re
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
import getpass

#%%
def download_matching_files(base_url, username=None, password=None, pattern=r"*.nc", dest_path="downloads"):
    """
    Download files matching the pattern from an HTTPS server with optional authentication.

    Handles:
    - Direct .nc links
    - THREDDS catalog links with ?dataset=... and converts them to fileServer URLs

    Parameters:
    -----------
    base_url : str
        URL of the server directory (e.g., THREDDS catalog URL)
    username : str or None
        Username for authentication (None or empty to skip auth)
    password : str or None
        Password for authentication (None or empty to skip auth)
    pattern : str
        Regex pattern to match filenames
    dest_path : str
        Destination directory path for downloaded files
    """
    # Create a session
    session = requests.Session()

    # Attach auth only if username/password provided
    if username and password:
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

        # Prepare THREDDS fileServer root if possible
        parsed_base = urlparse(base_url)
        file_server_root = f"{parsed_base.scheme}://{parsed_base.netloc}/thredds/fileServer/"

        candidates = []
        for a in soup.find_all('a'):
            href = a.get('href')
            if not href:
                continue

            parsed = urlparse(href)

            # Case 1: THREDDS catalog link with dataset query
            qs = parse_qs(parsed.query)
            dataset = None
            if 'dataset' in qs and qs['dataset']:
                dataset = qs['dataset'][0].lstrip('/')
                file_url = file_server_root + dataset
                file_name = os.path.basename(dataset)
            # Case 2: direct .nc link (absolute or relative)
            elif parsed.path and parsed.path.endswith('.nc'):
                if parsed.scheme:
                    file_url = href
                else:
                    file_url = urljoin(base_url, href)
                file_name = os.path.basename(urlparse(file_url).path)
            else:
                continue

            if file_name and re.match(pattern, file_name):
                candidates.append((file_name, file_url))

        if not candidates:
            print("No matching files found in directory listing.")
            return

        print(f"Found {len(candidates)} matching files.")

        # Download each matching file
        for file_name, file_url in candidates:
            local_path = os.path.join(dest_path, file_name)

            print(f"Downloading {file_name} from {file_url} ...")
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
    server_url = input("Enter server URL (e.g., https://example.com/data/ or THREDDS catalog URL): ")
    username = input("Enter username (leave blank if not required): ").strip() or None
    password = None
    if username:
        password = getpass.getpass("Enter password: ")
    dest_path = input("Enter destination directory path [downloads]: ") or "downloads"
    pattern = input("Enter file pattern e.g., [SICEv3.0_.*.nc]: ") or r"SICEv3.0_.*.nc"
    download_matching_files(server_url, username, password, pattern, dest_path)