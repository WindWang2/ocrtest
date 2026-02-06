"""
仪器读数识别系统 - LLM解析模块
支持规则解析、LM Studio、Ollama等多种后端
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举和数据类
# =============================================================================
class ValidationResult(Enum):
    """验证结果状态"""
    SUCCESS = "success"
    NEED_RETRY = "need_retry"
    FAILED = "failed"


@dataclass
class ParseResult:
    """解析结果"""
    status: ValidationResult
    data: Dict = field(default_factory=dict)
    confidence: float = 0.0
    message: str = ""
    missing_fields: List[str] = field(default_factory=list)
    retry_suggestion: str = ""
    raw_ocr_texts: List[str] = field(default_factory=list)


@dataclass
class InstrumentSchema:
    """仪器模式定义"""
    instrument_type: str
    name: str
    fields: Dict[str, Dict[str, Any]]
    keywords: List[str]
    required_fields: List[str] = field(default_factory=list)
    has_tabs: bool = False
    tabs: List[str] = field(default_factory=list)


# =============================================================================
# 加载仪器配置
# =============================================================================
def load_instrument_schemas() -> Dict[str, InstrumentSchema]:
    """从config.py加载仪器模式配置"""
    try:
        from config import INSTRUMENT_CONFIG
        
        schemas = {}
        for inst_type, config in INSTRUMENT_CONFIG.items():
            required = [
                f for f, v in config.get("fields", {}).items()
                if v.get("required", False)
            ]
            schemas[inst_type] = InstrumentSchema(
                instrument_type=inst_type,
                name=config.get("name", inst_type),
                fields=config.get("fields", {}),
                keywords=config.get("keywords", []),
                required_fields=required,
                has_tabs=config.get("has_tabs", False),
                tabs=config.get("tabs", []),
            )
        return schemas
    except ImportError:
        logger.warning("无法加载config.py，使用空配置")
        return {}


# 全局仪器模式
INSTRUMENT_SCHEMAS = load_instrument_schemas()


# =============================================================================
# LLM基类
# =============================================================================
class LLMBase(ABC):
    """LLM解析器基类"""
    
    # 可配置的阈值（降低以提高通过率）
    MIN_CONFIDENCE_THRESHOLD = 0.10
    MIN_TEXT_COUNT = 1
    REQUIRE_NUMBERS = False
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """生成文本响应"""
        pass
    
    def validate_ocr_results(
        self,
        ocr_results: List[Dict],
        instrument_type: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        验证OCR结果是否可用
        
        Args:
            ocr_results: OCR结果列表
            instrument_type: 仪器类型（可选）
        
        Returns:
            (是否通过, 原因说明)
        """
        if not ocr_results:
            return False, "OCR未识别到任何文字"
        
        if len(ocr_results) < self.MIN_TEXT_COUNT:
            return False, f"识别文本过少 ({len(ocr_results)}条)"
        
        # 计算置信度统计
        avg_conf = sum(r.get("confidence", 0) for r in ocr_results) / len(ocr_results)
        high_conf_count = sum(1 for r in ocr_results if r.get("confidence", 0) > 0.5)
        
        # 宽松策略：只要有高置信度结果或平均达到阈值就通过
        if avg_conf < self.MIN_CONFIDENCE_THRESHOLD and high_conf_count == 0:
            return False, f"OCR置信度过低 ({avg_conf:.2f})，建议增强对比度重试"
        
        # 可选：检查是否有数字
        if self.REQUIRE_NUMBERS:
            has_numbers = any(re.search(r'\d', r.get("text", "")) for r in ocr_results)
            if not has_numbers:
                return False, "未识别到任何数字，可能需要旋转图片"
        
        return True, "OCR结果可用"
    
    def identify_instrument_type(self, ocr_results: List[Dict]) -> Tuple[str, float]:
        """
        自动识别仪器类型
        
        Args:
            ocr_results: OCR结果列表
        
        Returns:
            (仪器类型, 匹配置信度)
        """
        if not INSTRUMENT_SCHEMAS:
            return "unknown", 0.0
        
        all_text = " ".join(r.get("text", "") for r in ocr_results)
        all_text_upper = all_text.upper()
        
        scores = {}
        for inst_type, schema in INSTRUMENT_SCHEMAS.items():
            score = 0
            
            # 关键词匹配
            for kw in schema.keywords:
                if kw.upper() in all_text_upper or kw in all_text:
                    score += 1
                    # 特征关键词加权
                    if kw.upper() in ["VISCOMETER", "LABHTD"]:
                        score += 2
                    elif kw in ["检测结果", "检测项目", "吸光度", "透光度"]:
                        score += 3
                    elif kw in ["表面张力", "Surface tension", "mN/m"]:
                        score += 3
                    elif kw.upper() in ["HUAZHI", "华志"]:
                        score += 2
            
            # 特殊模式匹配加分
            if inst_type == "electronic_scale":
                if re.search(r'\d+\.?\d*\s*g\b', all_text, re.I):
                    score += 5
            elif inst_type == "viscometer":
                if "速度" in all_text and "粘度" in all_text:
                    score += 4
            elif inst_type == "mixer_stirrer":
                if re.search(r'\d+\s*(rpm|转)', all_text, re.I):
                    score += 3
                if "高速" in all_text or "低速" in all_text:
                    score += 3
            elif inst_type == "surface_tensiometer":
                if re.search(r'\d+\.?\d*\s*mN/m', all_text):
                    score += 5
            elif inst_type == "thermo_hygrometer":
                if re.search(r'\d+\.?\d*\s*%\s*RH', all_text, re.I):
                    score += 4
            elif inst_type == "water_bath":
                if re.search(r'\d+\.?\d*\s*[°℃]', all_text):
                    score += 2
            elif inst_type == "ph_meter":
                if re.search(r'pH\s*[:：]?\s*\d+\.?\d*', all_text, re.I):
                    score += 5
            
            scores[inst_type] = score
        
        if scores and max(scores.values()) > 0:
            best_type = max(scores, key=scores.get)
            max_score = scores[best_type]
            total_keywords = len(INSTRUMENT_SCHEMAS[best_type].keywords)
            confidence = min(max_score / max(total_keywords * 0.3, 1), 1.0)
            return best_type, confidence
        
        return "unknown", 0.0
    
    def parse_instrument_reading(
        self,
        ocr_results: List[Dict],
        instrument_type: Optional[str] = None,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> ParseResult:
        """
        解析仪器读数
        
        Args:
            ocr_results: OCR结果列表
            instrument_type: 仪器类型（可选，自动识别）
            attempt: 当前尝试次数
            max_attempts: 最大尝试次数
        
        Returns:
            ParseResult对象
        """
        raw_texts = [r.get("text", "") for r in ocr_results]
        
        # 验证OCR结果
        is_valid, reason = self.validate_ocr_results(ocr_results, instrument_type)
        
        if not is_valid:
            if attempt < max_attempts:
                return ParseResult(
                    status=ValidationResult.NEED_RETRY,
                    message=reason,
                    retry_suggestion="increase_contrast" if "置信度" in reason else "try_rotation",
                    raw_ocr_texts=raw_texts,
                )
            else:
                return ParseResult(
                    status=ValidationResult.FAILED,
                    message=f"经过{max_attempts}次尝试仍无法识别: {reason}",
                    raw_ocr_texts=raw_texts,
                )
        
        # 自动识别仪器类型
        if instrument_type is None:
            instrument_type, type_confidence = self.identify_instrument_type(ocr_results)
            logger.info(f"自动识别仪器类型: {instrument_type} (置信度: {type_confidence:.2f})")
        
        schema = INSTRUMENT_SCHEMAS.get(instrument_type)
        
        # 构建prompt并调用LLM
        ocr_text = self._format_ocr_for_prompt(ocr_results)
        prompt = self._build_parsing_prompt(ocr_text, schema, instrument_type)
        response = self.generate(prompt)
        
        # 解析JSON
        result = self._extract_json_from_response(response)
        
        # 验证结果
        readings = result.get("readings", {})
        confidence = self._calculate_confidence(readings, schema, ocr_results)
        
        if not readings:
            if attempt < max_attempts:
                return ParseResult(
                    status=ValidationResult.NEED_RETRY,
                    message="未能提取到有效读数",
                    retry_suggestion="aggressive_preprocessing",
                    raw_ocr_texts=raw_texts,
                )
            else:
                return ParseResult(
                    status=ValidationResult.FAILED,
                    message="无法从OCR结果中提取有效数据",
                    data={"raw_ocr": raw_texts, "instrument_type": instrument_type},
                    raw_ocr_texts=raw_texts,
                )
        
        # 检查必要字段
        missing_fields = []
        if schema:
            for field_name in schema.required_fields:
                if field_name not in readings or readings[field_name] in [None, "", "N/A"]:
                    missing_fields.append(field_name)
        
        if missing_fields and attempt < max_attempts:
            return ParseResult(
                status=ValidationResult.NEED_RETRY,
                message=f"缺少必要字段: {missing_fields}",
                missing_fields=missing_fields,
                retry_suggestion="try_rotation",
                raw_ocr_texts=raw_texts,
            )
        
        # 成功
        result["instrument_type"] = instrument_type
        result["raw_ocr"] = raw_texts
        result["confidence"] = confidence
        
        return ParseResult(
            status=ValidationResult.SUCCESS,
            data=result,
            confidence=confidence,
            message="识别成功",
            raw_ocr_texts=raw_texts,
        )
    
    def _format_ocr_for_prompt(self, ocr_results: List[Dict]) -> str:
        """格式化OCR结果用于prompt"""
        lines = []
        for r in ocr_results:
            lines.append(f"- {r['text']} (置信度: {r.get('confidence', 0):.2f})")
        return "\n".join(lines)
    
    def _build_parsing_prompt(
        self,
        ocr_text: str,
        schema: Optional[InstrumentSchema],
        instrument_type: str,
    ) -> str:
        """构建解析prompt"""
        if schema:
            fields_desc = "\n".join(
                f"- {k} ({v.get('chinese', k)}): {v.get('type', 'string')}"
                + (" [必需]" if v.get('required') else "")
                + (f" (单位: {v.get('unit')})" if v.get('unit') else "")
                for k, v in schema.fields.items()
            )
            
            tabs_info = ""
            if schema.has_tabs:
                tabs_info = f"\n注意：该仪器有多个模式: {schema.tabs}，请识别当前模式。"
            
            prompt = f"""你是仪器数据解析助手。根据以下OCR结果提取仪器读数，输出JSON格式。

仪器类型: {instrument_type} ({schema.name})
{tabs_info}

需提取字段:
{fields_desc}

OCR结果:
{ocr_text}

请严格输出JSON格式:
{{
    "readings": {{"字段名": 值, ...}},
    "mode": "当前模式（如适用）",
    "confidence": 0.0-1.0
}}

规则：无法提取的字段设为null；数值字段只提取数字。只输出JSON。
"""
        else:
            prompt = f"""你是仪器数据解析助手。根据以下OCR结果提取读数，输出JSON格式。

仪器类型: {instrument_type}

OCR结果:
{ocr_text}

输出格式:
{{
    "readings": {{"参数名": 值, ...}},
    "confidence": 0.0-1.0
}}

只输出JSON。
"""
        return prompt
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """从响应中提取JSON"""
        # 直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 提取```json```块
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 提取{}块
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"无法提取JSON: {response[:200]}...")
        return {"raw_response": response, "parse_error": True}
    
    def _calculate_confidence(
        self,
        readings: Dict,
        schema: Optional[InstrumentSchema],
        ocr_results: List[Dict],
    ) -> float:
        """计算解析置信度"""
        if not readings:
            return 0.0
        
        # OCR平均置信度
        base_conf = (
            sum(r.get("confidence", 0) for r in ocr_results) / len(ocr_results)
            if ocr_results else 0.5
        )
        
        # 字段完整性
        if schema:
            total = len(schema.fields)
            filled = sum(1 for v in readings.values() if v not in [None, "", "N/A"])
            field_ratio = filled / total if total > 0 else 0.5
        else:
            field_ratio = 0.5 if readings else 0.0
        
        return min(max(base_conf * 0.4 + field_ratio * 0.6, 0.0), 1.0)


# =============================================================================
# 规则解析器
# =============================================================================
class RuleBasedParser(LLMBase):
    """基于规则的解析器（无需LLM）"""
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return ""
    
    def parse_instrument_reading(
        self,
        ocr_results: List[Dict],
        instrument_type: Optional[str] = None,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> ParseResult:
        """基于规则的解析"""
        raw_texts = [r.get("text", "") for r in ocr_results]
        
        # 验证
        is_valid, reason = self.validate_ocr_results(ocr_results, instrument_type)
        if not is_valid:
            if attempt < max_attempts:
                return ParseResult(
                    status=ValidationResult.NEED_RETRY,
                    message=reason,
                    retry_suggestion="increase_contrast",
                    raw_ocr_texts=raw_texts,
                )
            return ParseResult(
                status=ValidationResult.FAILED,
                message=f"经过{max_attempts}次尝试仍无法识别: {reason}",
                raw_ocr_texts=raw_texts,
            )
        
        # 识别仪器类型
        if instrument_type is None:
            instrument_type, _ = self.identify_instrument_type(ocr_results)
        
        schema = INSTRUMENT_SCHEMAS.get(instrument_type)
        
        # 规则提取
        raw_readings = self._extract_by_rules(ocr_results)
        mapped_readings = self._map_to_schema(raw_readings, schema)
        
        # 计算置信度
        confidence = self._calculate_confidence(mapped_readings or raw_readings, schema, ocr_results)
        
        if not mapped_readings and not raw_readings:
            if attempt < max_attempts:
                return ParseResult(
                    status=ValidationResult.NEED_RETRY,
                    message="未能提取到有效读数",
                    retry_suggestion="try_rotation",
                    raw_ocr_texts=raw_texts,
                )
            return ParseResult(
                status=ValidationResult.FAILED,
                message="无法提取有效数据",
                data={"raw_ocr": raw_texts, "instrument_type": instrument_type},
                raw_ocr_texts=raw_texts,
            )
        
        return ParseResult(
            status=ValidationResult.SUCCESS,
            data={
                "instrument_type": instrument_type,
                "readings": mapped_readings if mapped_readings else raw_readings,
                "raw_readings": raw_readings,
                "raw_ocr": raw_texts,
                "confidence": confidence,
            },
            confidence=confidence,
            message="识别成功",
            raw_ocr_texts=raw_texts,
        )
    
    def _extract_by_rules(self, ocr_results: List[Dict]) -> Dict:
        """基于规则提取数据"""
        readings = {}
        all_text = " ".join(r["text"] for r in ocr_results)
        
        # 模式匹配
        patterns = {
            "weight": (r'(\d+\.?\d*)\s*g\b', "g"),
            "temperature": (r'(\d+\.?\d*)\s*[°℃]', "℃"),
            "humidity": (r'(\d+\.?\d*)\s*%\s*(?:RH)?', "%RH"),
            "surface_tension": (r'(\d+\.?\d*)\s*mN/m', "mN/m"),
            "viscosity": (r'(\d+\.?\d*)\s*mPa', "mPa·s"),
            "current_rpm": (r'(\d+)\s*(?:转|rpm)', "rpm"),
            "density": (r'(\d+\.?\d*)\s*g/cm', "g/cm³"),
            "ph_value": (r'pH\s*[:：]?\s*(\d+\.?\d*)', ""),
        }
        
        for field_name, (pattern, unit) in patterns.items():
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                try:
                    readings[field_name] = float(match.group(1))
                    if unit:
                        readings[f"{field_name}_unit"] = unit
                except ValueError:
                    pass
        
        # 键值对提取
        for r in ocr_results:
            text = r["text"]
            for sep in [":", "：", "="]:
                if sep in text:
                    parts = text.split(sep, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = self._parse_value(parts[1].strip())
                        if key and value is not None:
                            readings[key] = value
                    break
        
        # 相邻元素配对
        for i in range(len(ocr_results) - 1):
            text = ocr_results[i]["text"].rstrip(":：=")
            next_text = ocr_results[i + 1]["text"]
            
            if self._is_label(text) and re.search(r'\d', next_text):
                value = self._parse_value(next_text)
                if value is not None:
                    readings[text] = value
        
        return readings
    
    def _map_to_schema(self, raw_readings: Dict, schema: Optional[InstrumentSchema]) -> Dict:
        """映射到标准字段"""
        if not schema:
            return raw_readings
        
        mapped = {}
        for field_name, field_info in schema.fields.items():
            chinese_name = field_info.get("chinese", "")
            
            if field_name in raw_readings:
                mapped[field_name] = raw_readings[field_name]
            else:
                for key, value in raw_readings.items():
                    if chinese_name and (chinese_name in key or key in chinese_name):
                        mapped[field_name] = value
                        break
        
        return mapped
    
    def _is_label(self, text: str) -> bool:
        """判断是否为标签"""
        if not text:
            return False
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        is_label = text.lower() in ["speed", "temp", "time", "rpm", "weight"]
        return (has_chinese or is_label) and not text[0].isdigit()
    
    def _parse_value(self, text: str) -> Union[float, str, None]:
        """解析数值"""
        if not text:
            return None
        
        match = re.search(r'([-+]?\d*\.?\d+)', text)
        if match:
            try:
                num = float(match.group(1))
                unit = text.replace(match.group(1), "").strip()
                return f"{num}{unit}" if unit and len(unit) < 10 else num
            except ValueError:
                pass
        
        return text if text else None


# =============================================================================
# LM Studio LLM
# =============================================================================
class LMStudioLLM(LLMBase):
    """LM Studio API后端"""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234",
        model: str = "local-model",
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import requests
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                },
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API错误: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            raise Exception(f"无法连接到LM Studio ({self.base_url})")


# =============================================================================
# Ollama LLM
# =============================================================================
class OllamaLLM(LLMBase):
    """Ollama API后端"""
    
    def __init__(
        self,
        model_name: str = "qwen2:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import requests
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise Exception(f"API错误: {response.text}")
        except requests.exceptions.ConnectionError:
            raise Exception(f"无法连接到Ollama ({self.base_url})")


# =============================================================================
# 工厂函数
# =============================================================================
def create_llm(model_type: str = "rule", **kwargs) -> LLMBase:
    """
    创建LLM实例
    
    Args:
        model_type: 类型 ("rule", "lmstudio", "ollama")
        **kwargs: 特定参数
    
    Returns:
        LLMBase实例
    """
    llms = {
        "rule": RuleBasedParser,
        "lmstudio": LMStudioLLM,
        "ollama": OllamaLLM,
    }
    
    if model_type not in llms:
        raise ValueError(f"不支持的LLM类型: {model_type}，可选: {list(llms.keys())}")
    
    return llms[model_type](**kwargs)
