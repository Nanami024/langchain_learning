from typing import List, Literal
from pydantic import BaseModel, Field

# 1. 第一步：从杂乱的新闻/反馈中提取出核心主体
class ExtractedNews(BaseModel):
    category: str = Field(description="行业或品类名称")
    core_content: str = Field(description="去掉干扰信息后的核心行业动态或消费者反馈内容")

# 2. 第二步：进行深度的趋势分析（符合实战作业要求）
class TrendInsight(BaseModel):
    category_name: str = Field(description="确定的品类名称")
    market_trend: Literal["上升", "下降", "平稳"] = Field(description="市场趋势倾向")
    consumer_pain_points: List[str] = Field(description="核心消费者痛点列表")
    churn_risk: bool = Field(description="是否存在严重的流失风险预警（True/False）")