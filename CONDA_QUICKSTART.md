# Conda 本地 Demo

Docker 暂时不需要。以下命令全部在 `researchflow-agent` 目录执行。

## 1. 创建环境

```powershell
& "C:\Users\lenovo\anaconda3\Scripts\conda.exe" env create -f environment.yml
```

已经创建后，只需在更新依赖时执行：

```powershell
& "C:\Users\lenovo\anaconda3\Scripts\conda.exe" env update -n researchflow -f environment.yml --prune
```

## 2. 启动后端

```powershell
.\scripts\start-backend.ps1
```

保持终端打开。API 地址：

- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/docs`

## 3. 启动前端

另开一个 PowerShell：

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173`。

## 4. Demo 流程

1. 上传文本型 PDF。
2. 等待论文状态变为可用。
3. 勾选论文并提问。
4. 在右侧检查页码引用与 `search_documents` 工具轨迹。
5. 选中一篇论文，通过数据库图标导入 `demo/experiment_metrics.csv`。
6. 提问“对比不同实验的 accuracy 和 f1 指标”。

项目运行固定使用 DeepSeek 对话模型和百炼 Embedding/Rerank。请复制 `.env.example` 为 `.env`，填写本地密钥后启动；自动化测试使用 `backend/tests/doubles.py`，不会调用付费接口。

