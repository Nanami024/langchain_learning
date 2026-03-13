import os
import json
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from schemas import ExtractedNews, TrendInsight 

# 1. 环境初始化
load_dotenv()

# 2. 实例化模型 - 显式开启 streaming=True
llm = ChatOpenAI(
    model="Pro/zai-org/GLM-4.7",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    temperature=0,
    streaming=True, # 开启流式支持
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 3. 初始化解析器
extraction_parser = JsonOutputParser(pydantic_object=ExtractedNews)
insight_parser = JsonOutputParser(pydantic_object=TrendInsight)

# 4. 定义 Prompt
extractor_prompt = ChatPromptTemplate.from_template(
    "你是一名行业数据整理专家。请从用户的输入中提取‘品类’和‘核心内容’。\n"
    "{format_instructions}\n"
    "用户输入内容：{user_input}"
)

analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深的行业分析师。请根据提取的信息分析市场趋势、痛点及风险。\n{format_instructions}"),
    ("human", "品类：{category}\n核心内容：{core_content}")
])

# 5. 构建 LCEL 链条
extractor_chain = (
    {
        "user_input": RunnablePassthrough(), 
        "format_instructions": lambda _: extraction_parser.get_format_instructions()
    } 
    | extractor_prompt 
    | llm 
    | extraction_parser
)

chain = (
    extractor_chain 
    | RunnablePassthrough.assign(
        format_instructions=lambda _: insight_parser.get_format_instructions()
    )
    | analyzer_prompt 
    | llm 
    | insight_parser
)

# ======================================================
# 6. 交互式实战与数据存储（流式改进版）
# ======================================================

history_records = []

print("\n" + "="*50)
print("【AI 品类趋势洞察抽取器 - 流式版】")
print("输入 'exit' 退出。")
print("="*50)

while True:
    user_input = input("\n请输入行业动态内容 > ").strip()
    
    if user_input.lower() == 'exit':
        print("\n" + "!"*10 + " 任务结束，正在保存记录 " + "!"*10)
        if history_records:
            with open("trend_insights_report.json", "w", encoding="utf-8") as f:
                json.dump(history_records, f, ensure_ascii=False, indent=4)
            print(f"✅ 成功记录 {len(history_records)} 条洞察结果，已保存至 'trend_insights_report.json'。")
        break
    
    if not user_input:
        continue

    try:
        print("\n>>> 正在深度分析中 (流式解析):")
        
        last_chunk = {}
        # 使用 .stream() 替代 .invoke()
        # 注意：由于是双链结构，第一步提取会静默完成，第二步分析会流式展示
        for chunk in chain.stream(user_input):
            last_chunk = chunk
            # 实时刷新当前解析出的 JSON 字段状态
            # 我们通过 sys.stdout 实现简单的原地刷新效果
            keys_found = list(chunk.keys())
            sys.stdout.write(f"\r已获取字段: {keys_found} ...")
            sys.stdout.flush()
        
        # 流式结束后，完整打印最终报告
        print("\n" + "-"*30)
        print(f"【品类名称】: {last_chunk.get('category_name', '解析中...')}")
        print(f"【市场趋势】: {last_chunk.get('market_trend', '解析中...')}")
        print(f"【痛点清单】: {', '.join(last_chunk.get('consumer_pain_points', []))}")
        print(f"【风险预警】: {'⚠️ 高风险' if last_chunk.get('churn_risk') else '✅ 正常'}")
        print("-" * 30)
        
        # 存入记录
        history_records.append({
            "input": user_input,
            "conclusion": last_chunk
        })
        
    except Exception as e:
        print(f"\n❌ 处理失败，错误原因: {e}")