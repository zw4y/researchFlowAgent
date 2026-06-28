# 10 React 工作台

## 学习目标

掌握 TypeScript 类型、组件状态、文件上传、SSE 解析、响应式布局和错误状态。

## 代码地图

- `frontend/src/api.ts`：API 与 SSE 客户端。
- `frontend/src/App.tsx`：论文、会话、消息、证据和工具状态。
- `frontend/src/styles.css`：三栏工作台和移动面板。

## 动手

在浏览器网络面板观察上传请求和 SSE 事件；切换到窄屏检查两侧抽屉。

## 练习

将大组件拆成 LibraryPanel、ChatPanel、EvidencePanel，并保持测试通过。

## 面试检查点

说明为什么 POST 流式对话使用 fetch + ReadableStream，而不是原生 EventSource。

