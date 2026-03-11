import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from schemas import *

# 1. 环境初始化：加载 .env 中的 API_KEY 和 BASE_URL
load_dotenv()

# 2. 实例化模型：使用 LangChain 的 ChatOpenAI 适配 SiliconFlow
llm = ChatOpenAI(
    model="Pro/zai-org/GLM-4.7",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    temperature=0,  # 分析任务建议设为 0，确保输出稳定
    streaming=True,
     model_kwargs={"response_format": {"type": "json_object"}}
)

# 3. 初始化解析器：将 Pydantic 模具注入解析器
parser = JsonOutputParser(pydantic_object=SentimentAnalysis)

# 4. 创建动态模板：使用 ChatPromptTemplate
# 注意：我们在这里动态注入了 format_instructions，告诉模型必须按 JSON 输出
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深的电商运营专家。请分析用户评论的情感，并严格按格式输出结果。\n{format_instructions}"),
    ("human", "产品品类：{category}\n用户评论内容：{review}")
])

# 5. LCEL 链式调用串联：这是 LangChain 的精髓
# 数据流向：输入 -> 提示词模板 -> 大模型 -> 解析器 -> 最终字典
chain = prompt | llm | parser

# 6. 实战测试：动态传入不同的变量
test_data = [
    {"category": "美妆", "review": "这瓶面霜质地很滋润，但是味道实在太难闻了，像塑料味。"},
    {"category": "数码", "review": "耳机音质炸裂！降噪效果非常好，坐地铁再也听不到噪音了，满分！"},
    {"category": "食品", "review": "快递挺快的，东西还没吃，先给个中评吧。"}
]

print(f"{'情感':<10} | {'评分':<5} | {'摘要'}")
print("-" * 50)

for data in test_data:
    # 动态传入变量，执行链条
    # 注意：必须传入 format_instructions 让解析器生效
    try:
        result = chain.invoke({
            "category": data["category"],
            "review": data["review"],
            "format_instructions": parser.get_format_instructions()
        })
        
        # 此时 result 已经是一个干净的 Python 字典了
        print(f"{result['sentiment']:<10} | {result['score']:<5} | {result['summary']}")
        
    except Exception as e:
        print(f"处理失败: {e}")