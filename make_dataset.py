from faker import Faker
import random
Faker.seed(0)
import requests
from tqdm import tqdm
import os
# set your API key in .env file
from dotenv import load_dotenv
load_dotenv()


MAX_REQUESTS = 1000
SIZE = 640 # Max 640x640
THRESHOLD = 5 * 1024  # 5KB
API_KEY = os.getenv("API_KEY")

image_dir = "data/coord_images_test"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)



fake = Faker()

coordinates = []
for i in range(MAX_REQUESTS):
    latlon = fake.location_on_land()  # (latitude, longitude, place name, two-letter country code, timezone)
    lat = str(float(latlon[0]) + round(random.uniform(-0.5, 0.5), 5))
    lon = str(float(latlon[1]) + round(random.uniform(-0.5, 0.5), 5))

    coordinates.append((round(float(lon), 5), round(float(lat), 5)))


# Start your session
session = requests.Session()

for i, coord in enumerate(tqdm(coordinates)):

    lon, lat = coord
    image_path = f"{image_dir}/{lat},{lon}.jpg"
    
    if os.path.exists(image_path):
        continue

    # Google Maps Static API's URL
    url = f"https://maps.googleapis.com/maps/api/streetview?size={SIZE}x{SIZE}&location={lat},{lon}&fov=100&source=outdoor&key={API_KEY}"

    try:
        response = session.get(url)
        
        # Save the image if it's larger than the threshold
        if len(response.content) > THRESHOLD:
            with open(image_path, "wb") as f:
                f.write(response.content)

    except Exception as e:
        tqdm.write(f"Error fetching image for {lat},{lon}: {e}")

# Close the session
session.close()