from fastapi import Header, HTTPException, status

from . import config


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not config.API_KEY:
        return
    if authorization != f"Bearer {config.API_KEY}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
