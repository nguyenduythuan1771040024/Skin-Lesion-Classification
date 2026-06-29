import os
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
CLASS_TO_IDX = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}

# Targets
TARGETS = {
    'df': 600,
    'vasc': 573,
    'akiec': 398
}

QUERIES = {
    'df': 'diagnosis_3:"Dermatofibroma"',
    'vasc': 'diagnosis_2:"Benign soft tissue proliferations - Vascular" AND image_type:dermoscopic',
    'akiec': 'diagnosis_3:"Solar or actinic keratosis" AND image_type:dermoscopic'
}

OUTPUT_DIR = 'data/isic_supplementary'
METADATA_CSV = 'data/isic_supplementary_metadata.csv'
HAM_METADATA_CSV = 'skin-cancer-mnist-ham10000/HAM10000_metadata.csv'

def fetch_image_metadata_list(query_str, limit_needed, exclude_set):
    """
    Fetch image metadata list from ISIC Archive API, paginating using cursor
    until we have enough new/non-overlapping images.
    """
    url = f'https://api.isic-archive.com/api/v2/images/search/?query={urllib.parse.quote(query_str)}&limit=100'
    valid_items = []
    
    while url and len(valid_items) < limit_needed:
        print(f"Querying API: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
                results = data.get('results', [])
                for r in results:
                    isic_id = r.get('isic_id')
                    # Check if already in HAM10000
                    if isic_id in exclude_set:
                        continue
                    
                    # Extract download URL
                    files = r.get('files', {})
                    download_url = None
                    if 'full' in files:
                        download_url = files['full'].get('url')
                    elif 'medium' in files:
                        download_url = files['medium'].get('url')
                    
                    if download_url:
                        valid_items.append({
                            'image_id': isic_id,
                            'download_url': download_url
                        })
                url = data.get('next')
                time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            break
            
    print(f"Found {len(valid_items)} new images for query '{query_str}'")
    return valid_items[:limit_needed]

def download_single_image(item, dx):
    """
    Download a single image from S3, with retry logic and timeout.
    """
    url = item['download_url']
    image_id = item['image_id']
    save_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")
    
    # Retry logic
    retries = 5
    backoff = 1.5
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as response:
                with open(save_path, 'wb') as f:
                    f.write(response.read())
            return {
                'image_id': image_id,
                'lesion_id': f'ISIC_supp_{image_id}',
                'dx': dx,
                'image_path': save_path,
                'label_idx': CLASS_TO_IDX[dx],
                'status': 'success'
            }
        except Exception as e:
            if i == retries - 1:
                print(f"Failed to download {image_id} from {url} after {retries} attempts: {e}")
                return {'image_id': image_id, 'status': 'failed', 'error': str(e)}
            time.sleep(backoff ** i)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load HAM10000 metadata to exclude existing images
    if os.path.exists(HAM_METADATA_CSV):
        print(f"Loading HAM10000 metadata from {HAM_METADATA_CSV} to prevent overlap...")
        ham_df = pd.read_csv(HAM_METADATA_CSV)
        ham_ids = set(ham_df['image_id'].tolist())
    else:
        print("Warning: HAM10000 metadata not found. No overlap check will be performed.")
        ham_ids = set()
        
    all_selected_items = {}
    
    # Step 1: Fetch metadata for all classes
    for dx, target in TARGETS.items():
        print(f"\n--- Fetching metadata for {dx} (Target: {target}) ---")
        query_str = QUERIES[dx]
        # Fetch slightly more than target to account for any download failures
        items = fetch_image_metadata_list(query_str, target + 10, ham_ids)
        all_selected_items[dx] = items
        
    # Step 2: Download images in parallel
    downloaded_metadata = []
    
    for dx, items in all_selected_items.items():
        target = TARGETS[dx]
        print(f"\n--- Downloading images for {dx} (Target: {target}, Available: {len(items)}) ---")
        
        success_count = 0
        # Download concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(download_single_image, item, dx): item for item in items}
            for future in as_completed(futures):
                res = future.result()
                if res and res.get('status') == 'success':
                    downloaded_metadata.append({
                        'image_id': res['image_id'],
                        'lesion_id': res['lesion_id'],
                        'dx': res['dx'],
                        'image_path': res['image_path'],
                        'label_idx': res['label_idx']
                    })
                    success_count += 1
                    # Stop if we hit the target
                    if success_count >= target:
                        print(f"Reached target {target} for {dx}. Cancelling remaining downloads.")
                        # Cancel remaining futures if possible
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
        print(f"Successfully downloaded {success_count} images for {dx}")
        
    # Step 3: Save metadata CSV
    meta_df = pd.DataFrame(downloaded_metadata)
    meta_df.to_csv(METADATA_CSV, index=False)
    print(f"\nSaved supplementary metadata of {len(meta_df)} images to {METADATA_CSV}")

if __name__ == '__main__':
    main()
