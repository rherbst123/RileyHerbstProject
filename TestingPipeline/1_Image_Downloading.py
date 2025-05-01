import requests
import os
from urllib.parse import unquote
import time
import urllib.parse




#Download images
def download_images_from_urls(url_file, output_dir="downloaded_images"):

    
    os.makedirs(output_dir, exist_ok=True)
    
    
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    total = len(urls)
    print(f"Found {total} URLs to process")
    
    # Download each image
    for idx, url in enumerate(urls):
        try:
            print(f"Downloading image {idx+1}/{total}")
            
            # Get image from URL
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Extract filename from URL
                parsed_url = urllib.parse.urlparse(url)
                filename = os.path.basename(parsed_url.path)
                filename = unquote(filename)
                
                # If no filename or no extension, create one based on content type
                if not filename or '.' not in filename:
                    content_type = response.headers.get('Content-Type', '')
                    if 'jpeg' in content_type or 'jpg' in content_type:
                        ext = '.jpg'
                    elif 'png' in content_type:
                        ext = '.png'
                    else:
                        ext = '.jpg'  # default
                    filename = f"image{ext}"
                
                # Create indexed filename
                new_filename = f"{idx+0:04d}_{filename}"
                output_path = os.path.join(output_dir, new_filename)
                
                # Save the image
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ Saved: {new_filename}")
            else:
                print(f"× Failed to download: {url} (Status: {response.status_code})")
                
        except Exception as e:
            print(f"× Error downloading {url}: {str(e)}")
        
        # Small delay to be nice to servers
        time.sleep(0.5)
    
    print(f"Download complete. Processed {total} URLs.")

if __name__ == "__main__":
    url_file = "c:\\Users\\Riley\\Desktop\\SingleImage.txt"
    output_dir = "C:\\Users\\riley\\Documents\\GitHub\\RileyHerbstProject\\TestingPipeline\\BaseImages"
    download_images_from_urls(url_file, output_dir)