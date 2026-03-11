import os
from dotenv import load_dotenv # 1. 引入加载库
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 2. 加载 .env 文件中的所有变量
load_dotenv()

# 3. 实例化模型，通过 os.getenv 安全获取变量
# LangChain 的 ChatOpenAI 会优先寻找名为 OPENAI_API_KEY 的变量
# 但因为你用的是 SiliconFlow，我们需要显式传递其 Base_url
llm = ChatOpenAI(
    model="Pro/zai-org/GLM-4.7",
    # 从环境变量获取，代码里不出现具体的 Key 字符串
    api_key=os.getenv("api_key"), 
    base_url=os.getenv("base_url"),
    temperature=0,
    streaming=True
)

# --- 以下进入任务 3 的业务逻辑 ---

# 定义模具
class TrendAnalysis(BaseModel):
    category: str = Field(description="品类名称")
    trend: str = Field(description="趋势描述")

parser = JsonOutputParser(pydantic_object=TrendAnalysis)

# 定义模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个行业分析专家。{format_instructions}"),
    ("human", "分析此段文本：{text}")
])

# LCEL 链条
chain = prompt | llm | parser

# 执行
result = chain.invoke({
    "text": "最近字节跳动旗下的精品咖啡业务增长迅速...",
    "format_instructions": parser.get_format_instructions()
})

print(result)