import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
def root():
    return {"status": "running"}


@app.post("/")
def extract(req: InvoiceRequest):
    try:

        prompt = f"""
You are an invoice extraction system.

Extract information from the invoice text.

Return ONLY valid JSON.

The JSON MUST satisfy this schema:

{json.dumps(req.schema, indent=2)}

Invoice Text:

{req.text}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        result = json.loads(response.text)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))