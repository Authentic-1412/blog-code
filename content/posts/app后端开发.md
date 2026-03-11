---
date : '2026-03-04T21:33:10+08:00'
draft : false
title : 'App后端开发'
tags: ["后端"]
categories: []
---
# 什么是API？
- 前后端工作的一个完整的工作流，是前端发送HTTP请求，后端接受，完整请求任务（查找信息、调用ai等），并返回请求值。由于一个应用功能很多，故给后端上需要分而治之，即每个功能定义一个api，实现：
    - 接收
    - 调用对应脚本（通常是许多函数，链接到对应脚本）
    - 返回
- 这个过程中，前端通过api定义的格式来发送请求到对应功能的api，并按照api定义的格式返回数据
# 如何搭建API？
- 自己从零开始写是不可能的，涉及许多HTTP等网络相关的繁琐工作，故需要**API框架**。这些API框架帮你完成了诸如通信、交互等繁琐的任务，你只需要专注于功能的实现即可。
- 常用的api框架有fastapi、flask、nodejs等，由于我对python比较熟悉，这里以fastapi为例。
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    # 不在路径大括号里的参数，会被 FastAPI 自动识别为 Query Parameter（查询参数）
    return {"item_id": item_id, "q": q}
```
- 以上是FASTAPI官方例程，这里做简单语法解释：
1. `@app.get("/")`
    - `@`是python的装饰器，我将其理解为api的标志
    - `get`：fastapi的方法，表示查看数据的**动作**（不更改数据），除此之外还有`post`（创建）、`put`（修改）、`delect`（删除）
    - `"/"`：路径。我觉得这是最抽象的一点。这个路径和你开发时app的目录结构没有任何关系，它代表的是你的**功能模块**之间的结构和路径，表示你要对哪个功能进行操作。举个例子，我想实现一个集查询信息和论坛功能一体的应用：
![example](/image/example.jpg)
    - 有关竞赛的功能：放在competitions目录下；有关论坛的，放在forum目录下，功能结构非常清晰
    - **总得来说，get等动作命令和路径化的功能信息明确指出了“要对哪个功能进行什么操作”。**
2. 下面的就是通过不同函数的调用实现功能并返回值了。

# 如何阅读API文档？（以FASTAPI为例）

# 部署代码到服务器上

## 推送到github
### 生成requirements.txt
- 使用`pipreqs`包，检测项目中用到的所有库和版本，并写入`requirements.txt`
> pip install pipreqs
> pipreqs . --force
- `.`: 当前目录（终端打开的目录）下所有文件的依赖库
- `--force`: 强制覆盖，即使已存在requirements文件

## 内网穿透
- 使用cpolar
0. `uvicorn app.main:app --host 0.0.0.0 --port 8080`开启你的后端
1. 下载、注册cpolar
2. 在“验证”中找到你的Authtoken
3. 在cmd上运行`cpolar authtoken <你的TOKEN内容>`，将你的token添加到你的环境变量中
4. cmd上`cpolar http 8080`
    - 输出一个http开头的随机域名，对应localhost。
    - 访问api：直接在该随机域名后添加api路径
    - FASTAPI文档：直接在该随机域名后添加/docs
- 免费模式，每次重启后端域名都会随机生成，下次开启后端后要重复步骤4，获取新域名。
