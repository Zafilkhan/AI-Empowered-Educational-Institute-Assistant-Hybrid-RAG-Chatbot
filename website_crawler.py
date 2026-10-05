import requests
from bs4 import BeautifulSoup
import os
import urllib3
from urllib.parse import urljoin

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://nitkkr.ac.in"

response = requests.get(url, verify=False)

soup = BeautifulSoup(response.text, "html.parser")

links = soup.find_all("a")

pdf_folder = "data/pdfs"

os.makedirs(pdf_folder, exist_ok=True)

for link in links:
    href = link.get("href")

    if href and ".pdf" in href:

        pdf_url = urljoin(url, href)

        print("Downloading:", pdf_url)

        pdf_data = requests.get(pdf_url, verify=False)

        filename = pdf_url.split("/")[-1]

        with open(os.path.join(pdf_folder, filename), "wb") as f:
            f.write(pdf_data.content)