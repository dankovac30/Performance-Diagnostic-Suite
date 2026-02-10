import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    "Central suite configuration"

    # 1080 Motion API
    MOTION_API_KEY = os.getenv("MOTION_1080_API_KEY")

    MOTION_BASE_URL = "https://publicapi.1080motion.com"
    MOTION_CLIENT_ENDPOINT = "/Client"
    MOTION_SESSIONS_ENDPOINT = "/Session/Search"
    MOTION_TRAINING_DATA_ENDPOINT = "/TrainingData/Session/"

    # VALD API
    VALD_CLIENT_ID = os.getenv("VALD_CLIENT_ID")
    VALD_TEAM_UID = os.getenv("VALD_TEAM_UID")
    VALD_CLIENT_SECRET = os.getenv("VALD_CLIENT_SECRET")
    VALD_TENANT_ID = os.getenv("VALD_TENANT_ID")

    VALD_TOKEN_URL = "https://security.valdperformance.com/connect/token"
    VALD_CLIENTS_URL = "https://prd-euw-api-externalprofile.valdperformance.com/profiles"

    VALD_FORCEDECKS_BASE_URL = "https://prd-euw-api-extforcedecks.valdperformance.com"
    VALD_FORCE_FRAME_BASE_URL = "https://prd-euw-api-externalforceframe.valdperformance.com"
    VALD_NORDBORD_BASE_URL = "https://prd-euw-api-externalnordbord.valdperformance.com"
    VALD_TESTS_ENDPOINT = "/tests"
    VALD_TESTS_ENDPOINT_V2 = "/tests/v2"
    VALD_TRIALS_ENDPOINT = "/v2019q3/teams/{team_uid}/tests/{test_id}/trials"
