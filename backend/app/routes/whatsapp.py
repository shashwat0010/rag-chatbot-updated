import logging
import re
from typing import Optional
from fastapi import APIRouter, Request, Response, Form, Query, BackgroundTasks, HTTPException
import httpx

from app.config import get_settings
from rag.pipeline import RAGPipeline
from services.guardrails import check_query_safety, DISCLAIMER
from models.schemas import QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])

_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

def format_for_whatsapp(response: QueryResponse) -> str:
    """Format RAG response into WhatsApp friendly plain text formatting."""
    # Convert Markdown bold **text** to WhatsApp bold *text*
    answer = response.answer
    answer = re.sub(r'\*\*(.*?)\*\*', r'*\1*', answer)
    
    def build_message(include_note: bool) -> str:
        parts = [answer]
        
        if response.citations:
            parts.append("\n*Citations:*")
            for i, citation in enumerate(response.citations, 1):
                year_str = f" ({citation.year})" if citation.year else ""
                parts.append(f"{i}. *{citation.title}* - {citation.journal}{year_str}\n   {citation.pubmed_url}")
                
        if include_note and response.confidence_note:
            note = re.sub(r'\*\*(.*?)\*\*', r'*\1*', response.confidence_note)
            parts.append(f"\n_Note: {note}_")
            
        return "\n".join(parts)

    # First try with note
    res_str = build_message(include_note=True)
    if len(res_str) > 1600:
        # Try without note to avoid truncation of core findings
        res_str = build_message(include_note=False)
        
    if len(res_str) > 1600:
        # Still too long, truncate with a notice
        notice = "\n\n_[Note: Message truncated due to WhatsApp length limits]_"
        res_str = res_str[:1600 - len(notice)] + notice
        
    return res_str

async def get_rag_reply(query: str) -> str:
    """Invokes safety checks and RAG pipeline to get formatted reply text."""
    settings = get_settings()
    if not settings.mistral_api_key:
        return "System configuration error: Mistral API key is missing."
        
    try:
        safety = await check_query_safety(query, settings.block_emergency_keywords)
        if not safety.allowed:
            return safety.message or "Query classification safety refusal."
            
        if safety.risk_level in ("NON_MEDICAL", "PATIENT_SPECIFIC") and safety.message:
            return safety.message

        # Run RAG (limit to 3 papers for mobile readability)
        result = await get_pipeline().run(
            query,
            max_papers=3,
            risk_level=safety.risk_level,
            analysis=safety
        )
        if not result.confidence_note.startswith("Low confidence"):
            result.confidence_note = f"{result.confidence_note} {DISCLAIMER}"
            
        return format_for_whatsapp(result)
    except Exception as e:
        logger.exception("Failed to run RAG pipeline for WhatsApp")
        return "An error occurred while searching the medical literature database. Please try again later."

# --- Twilio Webhook Handler ---

async def send_twilio_reply(to_number: str, from_number: str, text: str):
    """Asynchronous Twilio reply helper (sent via REST API using background task)"""
    settings = get_settings()
    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        logger.error("Twilio credentials not configured. Cannot send async reply.")
        return
        
    url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json"
    auth = (settings.twilio_account_sid, settings.twilio_auth_token)
    data = {
        "From": from_number,
        "To": to_number,
        "Body": text
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, auth=auth, data=data)
            r.raise_for_status()
            logger.info("Sent Twilio WhatsApp message response to %s", to_number)
        except Exception as e:
            if 'r' in locals() and hasattr(r, 'text'):
                logger.error("Twilio response status: %s | body: %s", r.status_code, r.text)
            logger.exception("Failed to send Twilio WhatsApp reply: %s", e)

async def process_and_reply_twilio(query: str, to_number: str, from_number: str):
    """Processes RAG query and sends reply via Twilio REST API"""
    try:
        reply_text = await get_rag_reply(query)
        await send_twilio_reply(to_number=to_number, from_number=from_number, text=reply_text)
    except Exception as e:
        logger.exception("Error in Twilio background task processing: %s", e)

@router.post("/twilio")
async def twilio_webhook(
    background_tasks: BackgroundTasks,
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(...)
):
    """Twilio WhatsApp Webhook POST handler"""
    logger.info("Received Twilio WhatsApp webhook from %s: %s", From, Body[:50])
    
    settings = get_settings()
    # Check if we should use REST API (async background tasks) or synchronous TwiML
    if settings.twilio_account_sid and settings.twilio_auth_token:
        # Async response (Avoids 15-second gateway timeout on slow RAG queries)
        background_tasks.add_task(process_and_reply_twilio, query=Body, to_number=From, from_number=To)
        return Response(content="", status_code=200)
    else:
        # Synchronous reply fallback (TwiML XML response)
        reply_text = await get_rag_reply(Body)
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{reply_text}</Message>
</Response>"""
        return Response(content=xml_content, media_type="application/xml")

# --- Meta WhatsApp Business Cloud API Webhook Handler ---

@router.get("/meta")
async def verify_meta_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """Meta webhook verification handler (GET)"""
    settings = get_settings()
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        return Response(content=hub_challenge, media_type="text/plain")
    logger.warning("Meta webhook verification failed. Token mismatch.")
    raise HTTPException(status_code=403, detail="Verification token mismatch")

async def send_meta_reply(phone_number_id: str, to_number: str, text: str):
    """Sends reply using Meta WhatsApp Graph API"""
    settings = get_settings()
    if not settings.whatsapp_token:
        logger.error("Meta WhatsApp Token not configured. Cannot send reply.")
        return
        
    url = f"https://graph.facebook.com/v20.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text
        }
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            logger.info("Sent Meta WhatsApp message response to %s", to_number)
        except Exception as e:
            logger.exception("Failed to send Meta WhatsApp reply: %s", e)

async def process_and_reply_meta(query: str, phone_number_id: str, to_number: str):
    """Processes RAG query and sends reply to Meta Business API"""
    try:
        reply_text = await get_rag_reply(query)
        await send_meta_reply(phone_number_id=phone_number_id, to_number=to_number, text=reply_text)
    except Exception as e:
        logger.exception("Error in Meta background task processing: %s", e)

@router.post("/meta")
async def meta_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Meta webhook event receiver (POST)"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    try:
        entry = body.get("entry", [])
        if not entry:
            return {"status": "no entry"}
            
        changes = entry[0].get("changes", [])
        if not changes:
            return {"status": "no changes"}
            
        value = changes[0].get("value", {})
        messages = value.get("messages", [])
        
        if not messages:
            return {"status": "no messages"}
            
        message = messages[0]
        msg_type = message.get("type")
        
        if msg_type != "text":
            return {"status": "ignored non-text message"}
            
        query = message.get("text", {}).get("body")
        sender = message.get("from")
        phone_number_id = value.get("metadata", {}).get("phone_number_id")
        
        if not query or not sender or not phone_number_id:
            return {"status": "missing metadata"}
            
        logger.info("Meta WhatsApp message from %s (phone_id=%s): %s", sender, phone_number_id, query[:50])
        
        # Enqueue background task to process RAG and send reply asynchronously
        background_tasks.add_task(
            process_and_reply_meta,
            query=query,
            phone_number_id=phone_number_id,
            to_number=sender
        )
        
    except Exception as e:
        logger.exception("Failed parsing Meta webhook message: %s", e)
        
    return {"status": "ok"}
