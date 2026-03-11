# 文件名：schemas.py
from typing import List, Literal
from pydantic import BaseModel, Field

class SentimentAnalysis(BaseModel):
    sentiment: Literal["正面", "负面", "中性"] = Field(description="用户的情感倾向")
    summary: str = Field(description="一句简短的评论摘要")
    keywords: List[str] = Field(description="核心关键词")
    score: int = Field(description="情绪评分，1-10分")