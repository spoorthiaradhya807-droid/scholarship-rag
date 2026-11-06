import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
OCR_MODEL = os.getenv("OCR_MODEL", GEN_MODEL)

RAW_DIR = BASE_DIR / "data" / "raw"
OCR_DIR = BASE_DIR / "data" / "processed" / "ocr"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
MANIFEST_PATH = BASE_DIR / "data" / "manifest.csv"
EMBED_DIMENSIONS = 768
COLLECTION_NAME = "scholarships"
TOP_K = 6                  # chunks retrieved per question
RELEVANCE_MIN = 0.60       # below this similarity, refuse without calling Gemini
HISTORY_TURNS = 3          # earlier exchanges used to understand follow-ups
MAX_QUESTION_CHARS = 1000
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
MAX_UPLOAD_MB = 10
SHOW_PAGE_NUMBERS = False
SHOW_CITATION_NUMBERS = False
LIST_TOP_K = 20            # chunks retrieved when the student asks for a list
LIST_RELEVANCE_MIN = 0.50
SHOW_SCREENSHOT_NOTE = False