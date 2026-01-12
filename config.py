import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    "Central suite configuration"

    # 1080 Motion API
    MOTION_API_KEY = os.getenv("MOTION_1080_API_KEY")
