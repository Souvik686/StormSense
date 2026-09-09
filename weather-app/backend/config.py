import os
from dotenv import load_dotenv

# Load variables from the project's .env file
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise RuntimeError(
        "OPENWEATHER_API_KEY is missing. "
        "Check the .env file in the Weather Nowcasting project root."
    )