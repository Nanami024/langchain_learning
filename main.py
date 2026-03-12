import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from schemas import *

# 1. 环境初始化
load_dotenv()

# 2. 实例化模型
llm = ChatOpenAI(
    model="Pro/zai-org/GLM-4.7",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    temperature=0,
    streaming=True,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 3. 初始化解析器
extraction_parser = JsonOutputParser(pydantic_object=ExtractedReview)
parser = JsonOutputParser(pydantic_object=SentimentAnalysis)

# 4. 定义 Prompt
extractor_prompt = ChatPromptTemplate.from_template(
    "你是一个电商数据整理专家。请从用户的输入中提取‘产品品类’和‘评论内容’。\n"
    "{format_instructions}\n"
    "用户输入内容：{user_input}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深的电商运营专家。请分析用户评论的情感，并严格按格式输出结果。\n{format_instructions}"),
    ("human", "产品品类：{category}\n用户评论内容：{review}")
])

# ======================================================
# 5. 使用 RunnablePassthrough 串联双链
# ======================================================

# 提取链：输入字符串 -> 结构化字典 {"category": "...", "review": "..."}
extractor_chain = (
    {
        "user_input": RunnablePassthrough(), 
        "format_instructions": lambda _: extraction_parser.get_format_instructions()
    } 
    | extractor_prompt 
    | llm 
    | extraction_parser
)

# 最终全自动链
# 流程：提取 -> 复制提取结果并补上情感分析指令 -> 情感分析
chain = (
    extractor_chain 
    | RunnablePassthrough.assign(
        format_instructions=lambda _: parser.get_format_instructions()
    )
    | prompt 
    | llm 
    | parser
)

# ======================================================
# 6. 交互式实战测试
# ======================================================

print("\n" + "="*50)
print("【AI 电商评论智能分析系统】")
print("直接输入评论（如：'这件衣服质量太差了'），系统会自动识别品类并分析。")
print("输入 'exit' 退出程序。")
print("="*50)

while True:
    user_input = input("\n请输入评论内容 > ").strip()
    
    if user_input.lower() == 'exit':
        print("程序已退出。")
        break
    
    if not user_input:
        continue

    try:
        # 只需要传入原始字符串，RunnablePassthrough 会处理剩下的逻辑
        result = chain.invoke(user_input)
        
        # 输出结果
        print("-" * 30)
        print(f"【情感反馈】: {result['sentiment']}")
        print(f"【情绪评分】: {result['score']} / 10")
        print(f"【内容摘要】: {result['summary']}")
        print(f"【关键词】: {', '.join(result['keywords'])}")
        print("-" * 30)
        
    except Exception as e:
        print(f"处理失败，错误原因: {e}")