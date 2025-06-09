import pathlib, requests, zipfile, io

RAW_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "wine_quality.zip"

    if zip_path.exists():
        print("Archive already downloaded")
    else:
        print(f"Downloading {URL} …")
        r = requests.get(URL, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        print("Download complete")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    print("Extraction done")

if __name__ == "__main__":
    main()