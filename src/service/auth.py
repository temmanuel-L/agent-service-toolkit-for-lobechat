"""
认证和中间件模块
"""
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core import settings


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    """
    验证HTTP Bearer Token认证

    该函数用于验证请求中的Bearer Token是否有效如果未提供Token或Token无效，则抛出401未授权异常
    如果未配置AUTH_SECRET，则跳过验证

    Args:
        http_auth (Annotated[HTTPAuthorizationCredentials | None]): 从请求头中获取的认证信息

    Raises:
        HTTPException: 当认证失败时抛出401未授权异常
    """

    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def custom_generate_unique_id(route) -> str:
    """
    生成路由的唯一标识符

    该函数用于为FastAPI路由生成自定义的唯一ID，这里直接使用路由的名称作为唯一标识符
    这样可以简化路由操作ID的生成，使其更直观易读

    Args:
        route: FastAPI路由对象，包含路由的相关信息

    Returns:
        str: 路由名称字符串，用作该路由的唯一标识符
    """
    return route.name