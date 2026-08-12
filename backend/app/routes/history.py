import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from models.database import SearchHistory, User, get_db
from models.schemas import SearchHistoryItem
from services.auth import require_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["history"])


@router.get("/history", response_model=List[SearchHistoryItem])
@router.get("/api/history", response_model=List[SearchHistoryItem])
async def get_user_search_history(
    current_user: User = Depends(require_current_user),
    db: Session = Depends(get_db)
):
    searches = db.query(SearchHistory)\
        .filter(SearchHistory.user_id == current_user.id)\
        .order_by(SearchHistory.created_at.desc())\
        .all()
    
    res = []
    for s in searches:
        res.append(SearchHistoryItem(
            id=s.id,
            query=s.query,
            response=s.response,
            confidence_score=s.confidence_score,
            evidence_count=s.evidence_count,
            created_at=s.created_at.isoformat() if s.created_at else ""
        ))
    return res


@router.delete("/history/{history_id}")
@router.delete("/api/history/{history_id}")
async def delete_search_item(
    history_id: int,
    current_user: User = Depends(require_current_user),
    db: Session = Depends(get_db)
):
    item = db.query(SearchHistory)\
        .filter(SearchHistory.id == history_id, SearchHistory.user_id == current_user.id)\
        .first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History item not found"
        )
    
    db.delete(item)
    db.commit()
    return {"status": "deleted", "id": history_id}
