-- =============================================================================
-- S&OP 决策中枢 v6.0 —— MySQL 初始化脚本（**可选**）
--
-- 说明：
--   1. 你**不一定**要执行这份 SQL；运行 `python import_csv_to_mysql.py --drop`
--      会自动完成「建库 + 建 sales_performance 表 + 灌 1000 行数据」。
--   2. `chat_messages` 表由 LangChain `SQLChatMessageHistory` 在前端第一次发消息时
--      自动 CREATE，无需手动建。本文件只把 DDL 写出来供你审阅与备案。
--   3. 用 Navicat / DBeaver / mysql CLI 都可以执行。如果想把每段拆开跑，按 `--`
--      分组复制即可。
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1) 建库
-- -----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS `sop_ai_system`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `sop_ai_system`;

-- -----------------------------------------------------------------------------
-- 2) （强烈推荐）单独建一个只读账号给 SQL Agent
--    填好密码后取消注释执行；并把它写到 w6/.env 的 MYSQL_RO_USER / MYSQL_RO_PASSWORD。
--    没建也没关系：sql_agent_tool.py 会回退到 MYSQL_USER，并依赖 SQLAlchemy 兜底拦截。
-- -----------------------------------------------------------------------------
-- CREATE USER 'sop_ro'@'%' IDENTIFIED BY '把这里改成你的密码';
-- GRANT SELECT ON `sop_ai_system`.* TO 'sop_ro'@'%';
-- FLUSH PRIVILEGES;

-- -----------------------------------------------------------------------------
-- 3) 业务事实表：S&OP 门店日度销售与考核明细
--    与 `import_csv_to_mysql.py` 中的 DDL 保持一致；列含义详见列注释。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `sales_performance` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `biz_date` DATE NOT NULL COMMENT '业务日期（原 CSV 的「日期」）',
  `category` VARCHAR(64) NOT NULL COMMENT '商品类目（家居百货/食品/服装鞋靴/化妆品 等）',
  `region` VARCHAR(32) NOT NULL COMMENT '所属区域（华东/华北/华南/华中/西南，演示合成）',
  `store_code` VARCHAR(32) NOT NULL COMMENT '门店编号（区域前缀+三位序号，如 EC001、NC003）',
  `forecast_qty` INT NOT NULL COMMENT '预测销量',
  `actual_qty` INT NOT NULL COMMENT '实际销量',
  `forecast_accuracy` DECIMAL(5,2) NOT NULL COMMENT '预测准确率（%）',
  `oos_rate` DECIMAL(5,2) NOT NULL COMMENT '缺货率（%）',
  `inventory_turnover_days` INT NOT NULL COMMENT '库存周转天数',
  PRIMARY KEY (`id`),
  KEY `idx_date` (`biz_date`),
  KEY `idx_region` (`region`),
  KEY `idx_store` (`store_code`),
  KEY `idx_category` (`category`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='S&OP 门店日度销售与考核明细（v6.0 演示数据，含合成的区域与门店列）';

-- -----------------------------------------------------------------------------
-- 4) 会话历史表（LangChain SQLChatMessageHistory 默认会自动创建；列出供参考）
--    字段名固定为 id / session_id / message，不要改名（LangChain 内部硬编码）。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `chat_messages` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `session_id` TEXT COMMENT 'FastAPI 传入的会话 ID（前端 URL 中的 sid）',
  `message` TEXT COMMENT 'LangChain 序列化后的消息 JSON（含 role、content、tool_calls 等）',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='LangChain SQLChatMessageHistory 持久化会话历史';

-- -----------------------------------------------------------------------------
-- 5) 验证 SQL（导入数据后用 Navicat 跑这几条核对）
-- -----------------------------------------------------------------------------
-- SELECT COUNT(*) AS total_rows FROM sales_performance;                           -- 期望 1000
-- SELECT region, COUNT(*) FROM sales_performance GROUP BY region;                 -- 期望每个区域 200
-- SELECT MIN(biz_date), MAX(biz_date) FROM sales_performance;                     -- 2023-01-05 ~ 2026-03-xx
--
-- -- 演示问题：华东区 2026 年 3 月预测准确率最高的门店
-- SELECT store_code, ROUND(AVG(forecast_accuracy), 2) AS avg_acc
-- FROM sales_performance
-- WHERE region = '华东'
--   AND biz_date >= '2026-03-01' AND biz_date < '2026-04-01'
-- GROUP BY store_code
-- ORDER BY avg_acc DESC
-- LIMIT 5;
--
-- -- 持久化验证：列出所有会话与消息数
-- SELECT session_id, COUNT(*) AS messages
-- FROM chat_messages
-- GROUP BY session_id
-- ORDER BY MAX(id) DESC;
