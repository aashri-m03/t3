import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

client = genai.Client(api_key=API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvoiceRequest(BaseModel):
    document_id: str
    text: str
    schema: dict


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/")
def extract(req: InvoiceRequest):
    try:

        prompt = f"""
Extract the invoice information.

Return ONLY valid JSON.

The JSON MUST strictly follow this schema:

{json.dumps(req.schema)}

Invoice text:

{req.text}
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0
            )
        )

        if not response.text:
            raise Exception("Gemini returned an empty response.")

        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))