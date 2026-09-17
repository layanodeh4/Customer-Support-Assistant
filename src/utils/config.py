import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")


RAW_OLIST = ROOT / "data" / "raw" / "olist"

RAW_SUPPORT = ROOT / "data" / "raw" / "support"

PROCESSED = ROOT / "data" / "processed"

ARTIFACTS = ROOT / "artifacts"

POLICIES = ROOT / "docs" / "policies"

DB_PATH = ARTIFACTS / "ecommerce.db"


OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna"
)