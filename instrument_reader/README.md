# 仪器读数识别系统

基于OCR和大模型的实验室仪器显示屏数据自动识别系统。

## 📋 功能特点

- **多仪器支持**: 水质检测仪、电子秤、粘度计、混调器、表面张力仪、温湿度计等10+种仪器
- **智能识别**: 自动识别仪器类型，无需手动指定
- **自动重试**: OCR失败时自动调整预处理参数重试
- **多后端支持**: 支持规则解析、LM Studio、Ollama等多种LLM后端
- **详细输出**: 终端显示OCR识别结果、置信度统计、解析过程

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install opencv-python Pillow requests

# OCR引擎 (推荐PaddleOCR)
pip install paddlepaddle paddleocr
```

### 2. 基本使用

```bash
# 处理单张图片
python main.py -i photo.jpg

# 处理整个目录
python main.py -d ./images

# 输出到JSON文件
python main.py -d ./images -o results.json
```

### 3. 使用大模型

```bash
# 使用LM Studio (需先启动LM Studio并加载模型)
python main.py -i photo.jpg --llm lmstudio

# 使用Ollama
python main.py -i photo.jpg --llm ollama --ollama-model qwen2:7b
```

## 📁 项目结构

```
instrument_reader/
├── main.py           # 主程序入口
├── config.py         # 配置文件（仪器定义、阈值设置）
├── ocr_module.py     # OCR模块（PaddleOCR/EasyOCR）
├── llm_module.py     # LLM解析模块
├── requirements.txt  # 依赖清单
└── README.md         # 本文件
```

## 🔧 支持的仪器类型

| 类型ID | 中文名 | 关键特征 |
|--------|--------|----------|
| `water_quality_tester` | 水质检测仪 | 检测结果、吸光度、透光度 |
| `electronic_scale` | 电子秤 | 数字+g/kg、HUAZHI |
| `viscometer` | 粘度计 | VISCOMETER、速度、粘度 |
| `mixer_stirrer` | 混调器/搅拌器 | rpm、高速/低速 |
| `water_bath` | 恒温水浴锅 | 温度、℃ |
| `surface_tensiometer` | 表面张力仪 | mN/m |
| `thermo_hygrometer` | 温湿度计 | %RH |
| `ph_meter` | pH计 | pH值 |
| `conductivity_meter` | 电导率仪 | μS/cm |
| `do_meter` | 溶解氧仪 | mg/L |

## 📊 输出示例

### 终端输出

```
============================================================
📷 处理图片: im001.jpg
============================================================

🔄 尝试 1/3
   预处理策略: default

   📝 OCR识别结果 (8条):
  [ 1] █████████░ 0.92 │ 检测结果
  [ 2] ████████░░ 0.85 │ 检测项目
  [ 3] ████████░░ 0.83 │ 总硬度（低量程）
  [ 4] █████████░ 0.91 │ 吸光度
  [ 5] █████████░ 0.89 │ 0.000
  ...

   📊 置信度: 平均=0.87, 最低=0.75, 最高=0.92

   🤖 LLM解析中...

   ✅ 识别成功!
   仪器类型: 水质检测仪 (water_quality_tester)
   置信度: 85.00%
```

### JSON输出

```json
{
  "timestamp": "2026-02-05T12:00:00",
  "total": 9,
  "success": 8,
  "results": [
    {
      "success": true,
      "image_path": "im001.jpg",
      "instrument_type": "water_quality_tester",
      "readings": {
        "test_item": "总硬度（低量程）",
        "test_date": "2026-01-12 16:12:41",
        "absorbance": 0.0,
        "transmittance": "100.00%",
        "content": "0mg/L"
      },
      "confidence": 0.85
    }
  ]
}
```

## ⚙️ 配置说明

### 调整置信度阈值

编辑 `config.py`:

```python
OCR_RETRY_CONFIG = {
    "max_retries": 3,              # 重试次数
    "confidence_threshold": 0.10,  # 置信度阈值（降低可提高通过率）
}
```

或编辑 `llm_module.py` 中的类属性:

```python
class LLMBase(ABC):
    MIN_CONFIDENCE_THRESHOLD = 0.10  # 调整这里
```

### 添加新仪器

在 `config.py` 的 `INSTRUMENT_CONFIG` 中添加:

```python
"new_instrument": {
    "name": "新仪器名称",
    "keywords": ["关键词1", "关键词2"],
    "fields": {
        "field1": {"chinese": "字段1", "type": "number", "required": True},
        "field2": {"chinese": "字段2", "type": "string"},
    },
}
```

## 🔌 LLM后端配置

### LM Studio

1. 下载并安装 [LM Studio](https://lmstudio.ai/)
2. 加载一个模型（推荐Qwen2-7B或类似）
3. 启动本地服务器（默认端口1234）
4. 运行: `python main.py -i photo.jpg --llm lmstudio`

### Ollama

1. 安装 [Ollama](https://ollama.ai/)
2. 拉取模型: `ollama pull qwen2:7b`
3. 运行: `python main.py -i photo.jpg --llm ollama`

## 📝 命令行参数

```
usage: main.py [-h] (-i IMAGE | -d DIR) [--ocr {paddle,easyocr}]
               [--llm {rule,lmstudio,ollama}] [--lmstudio-url URL]
               [--ollama-model MODEL] [-t TYPE] [-o OUTPUT] [-r] [-q]

选项:
  -i, --image     单张图片路径
  -d, --dir       图片目录路径
  --ocr           OCR引擎 (paddle/easyocr)
  --llm           LLM类型 (rule/lmstudio/ollama)
  -t, --type      指定仪器类型
  -o, --output    输出JSON文件路径
  -r, --recursive 递归处理子目录
  -q, --quiet     安静模式
```

## 🐛 常见问题

### OCR识别率低

1. 确保图片清晰、光线充足
2. 降低 `confidence_threshold` 到 0.05
3. 尝试使用 `--ocr easyocr` 切换引擎

### 仪器类型识别错误

1. 在 `config.py` 中为该仪器添加更多关键词
2. 使用 `-t TYPE` 手动指定仪器类型

### LM Studio连接失败

1. 确保LM Studio已启动并加载了模型
2. 检查端口是否正确: `--lmstudio-url http://127.0.0.1:1234`

## 📄 License

MIT License
