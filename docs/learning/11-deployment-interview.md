# 11 部署与面试

## 学习目标

理解 Dockerfile、Compose、数据卷、反向代理、迁移、CI 和 README。

## 代码地图

- `compose.yaml`：四个服务和三个持久化卷。
- `backend/Dockerfile`、`frontend/Dockerfile`：多阶段构建。
- `frontend/nginx.conf`：API、SSE 与 MCP 代理。
- `.github/workflows/ci.yml`：前后端质量门禁。

## 动手

安装 Docker 后执行 `docker compose up --build`，删除容器再启动，确认论文元数据和向量仍存在。

## 练习

增加生产环境 HTTPS 入口，并为 API 添加非 root 容器用户。

## 面试检查点

准备项目一分钟介绍、架构图、一次真实故障、一次技术取舍和下一阶段扩展计划。

