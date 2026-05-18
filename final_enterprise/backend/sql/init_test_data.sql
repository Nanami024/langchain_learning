-- =========================================================
-- Agentic S&OP MySQL 初始化脚本（测试数据版）
-- 作用：
-- 1) 创建数据库与业务表
-- 2) 创建会话历史表
-- 3) （可选）创建只读账号并授权
-- 4) 写入可用于全链路演示的测试数据
-- =========================================================

CREATE DATABASE IF NOT EXISTS `sop_ai_system` DEFAULT CHARSET utf8mb4;
USE `sop_ai_system`;

-- 可选：创建只读账号（如已存在可忽略报错）
CREATE USER IF NOT EXISTS 'sop_ro'@'%' IDENTIFIED BY '15301385065a';
GRANT SELECT ON `sop_ai_system`.* TO 'sop_ro'@'%';
FLUSH PRIVILEGES;

-- 会话历史表（当前代码可写 role/content 结构）
CREATE TABLE IF NOT EXISTS `chat_messages` (
  `id` BIGINT PRIMARY KEY AUTO_INCREMENT,
  `session_id` VARCHAR(128) NOT NULL,
  `role` VARCHAR(32) NOT NULL,
  `content` TEXT NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 业务数据表
CREATE TABLE IF NOT EXISTS `sales_performance` (
  `id` BIGINT PRIMARY KEY AUTO_INCREMENT,
  `biz_date` DATE NOT NULL,
  `region` VARCHAR(32) NOT NULL,
  `store_code` VARCHAR(32) NOT NULL,
  `sku` VARCHAR(64) NOT NULL,
  `forecast_accuracy` DOUBLE NOT NULL,
  `oos_rate` DOUBLE NOT NULL,
  `inventory_turnover_days` DOUBLE NOT NULL,
  KEY `idx_date` (`biz_date`),
  KEY `idx_region` (`region`),
  KEY `idx_store` (`store_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 如需重置测试数据，先清空
TRUNCATE TABLE `sales_performance`;

-- =========================================================
-- 测试数据（覆盖 2026-03，便于验证“上个月”问题）
-- =========================================================
INSERT INTO `sales_performance`
(`biz_date`,`region`,`store_code`,`sku`,`forecast_accuracy`,`oos_rate`,`inventory_turnover_days`)
VALUES
('2026-03-01','华东','E-01','SKU-001',86.2,2.1,22.0),
('2026-03-01','华东','E-02','SKU-002',83.4,2.6,24.0),
('2026-03-01','华东','E-03','SKU-003',88.1,1.9,21.0),
('2026-03-01','华南','S-01','SKU-004',79.8,3.8,29.0),
('2026-03-01','华南','S-02','SKU-005',77.5,4.2,31.0),
('2026-03-01','华北','N-01','SKU-006',74.2,4.9,33.0),
('2026-03-01','华北','N-02','SKU-007',72.8,5.1,35.0),

('2026-03-08','华东','E-01','SKU-001',87.5,2.0,21.0),
('2026-03-08','华东','E-02','SKU-002',84.0,2.4,23.0),
('2026-03-08','华东','E-03','SKU-003',89.2,1.7,20.0),
('2026-03-08','华南','S-01','SKU-004',80.4,3.6,28.0),
('2026-03-08','华南','S-02','SKU-005',76.9,4.4,32.0),
('2026-03-08','华北','N-01','SKU-006',73.1,5.0,34.0),
('2026-03-08','华北','N-02','SKU-007',71.9,5.3,36.0),

('2026-03-15','华东','E-01','SKU-001',88.4,1.8,20.0),
('2026-03-15','华东','E-02','SKU-002',84.7,2.3,22.0),
('2026-03-15','华东','E-03','SKU-003',90.1,1.6,19.0),
('2026-03-15','华南','S-01','SKU-004',81.2,3.4,27.0),
('2026-03-15','华南','S-02','SKU-005',77.8,4.1,30.0),
('2026-03-15','华北','N-01','SKU-006',72.5,5.2,35.0),
('2026-03-15','华北','N-02','SKU-007',70.8,5.6,37.0),

('2026-03-22','华东','E-01','SKU-001',87.9,1.9,21.0),
('2026-03-22','华东','E-02','SKU-002',85.1,2.2,22.0),
('2026-03-22','华东','E-03','SKU-003',90.5,1.5,19.0),
('2026-03-22','华南','S-01','SKU-004',80.9,3.5,28.0),
('2026-03-22','华南','S-02','SKU-005',78.1,4.0,29.0),
('2026-03-22','华北','N-01','SKU-006',71.9,5.3,36.0),
('2026-03-22','华北','N-02','SKU-007',69.7,5.9,38.0),

('2026-03-29','华东','E-01','SKU-001',88.7,1.7,20.0),
('2026-03-29','华东','E-02','SKU-002',85.4,2.1,21.0),
('2026-03-29','华东','E-03','SKU-003',91.0,1.4,18.0),
('2026-03-29','华南','S-01','SKU-004',81.5,3.3,27.0),
('2026-03-29','华南','S-02','SKU-005',78.6,3.9,28.0),
('2026-03-29','华北','N-01','SKU-006',71.2,5.4,37.0),
('2026-03-29','华北','N-02','SKU-007',68.9,6.1,39.0),

-- 4 月少量数据（用于边界）
('2026-04-05','华东','E-01','SKU-001',86.8,2.1,22.0),
('2026-04-05','华东','E-02','SKU-002',84.2,2.6,24.0),
('2026-04-05','华南','S-01','SKU-004',79.6,3.9,30.0),
('2026-04-05','华北','N-01','SKU-006',70.5,5.8,38.0);

-- 快速检查
SELECT COUNT(*) AS total_rows FROM `sales_performance`;
SELECT region, ROUND(AVG(forecast_accuracy),2) AS avg_acc
FROM `sales_performance`
WHERE biz_date >= '2026-03-01' AND biz_date < '2026-04-01'
GROUP BY region
ORDER BY avg_acc DESC;
