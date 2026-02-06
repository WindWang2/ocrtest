"""
LLM模块 - 使用开源大模型将OCR结果结构化为JSON
增强版：支持结果验证、OCR重试请求和七段数码管小数点修正
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """验证结果枚举"""
    SUCCESS = "success"           # 识别成功
    NEED_RETRY = "need_retry"     # 需要重试OCR
    FAILED = "failed"             # 彻底失败


@dataclass
class ParseResult:
    """解析结果数据类"""
    status: ValidationResult
    data: Dict = field(default_factory=dict)
    confidence: float = 0.0
    message: str = ""
    missing_fields: List[str] = field(default_factory=list)
    retry_suggestion: str = ""
    raw_ocr_texts: List[str] = field(default_factory=list)  # 原始OCR文本列表
    
    def __post_init__(self):
        """初始化后处理：从data中提取raw_ocr_texts"""
        if not self.raw_ocr_texts and self.data:
            # 尝试从data中获取raw_ocr
            raw_ocr = self.data.get("raw_ocr", [])
            if raw_ocr:
                self.raw_ocr_texts = raw_ocr


@dataclass
class InstrumentSchema:
    """仪器数据模式定义"""
    instrument_type: str
    fields: Dict[str, Dict[str, Any]]
    keywords: List[str]
    required_fields: List[str] = field(default_factory=list)
    has_tabs: bool = False
    tabs: List[str] = field(default_factory=list)


# =====================================================
# 七段数码管小数点修正器
# =====================================================
class SegmentDisplayDecimalFixer:
    """
    七段数码管小数点修正器
    
    针对LED/LCD七段数码管显示的OCR识别问题：
    - 小数点可能未被识别（如 90.5 识别为 905，140.39 识别为 14039）
    - 根据仪器类型和数值范围智能修正
    
    主要适用场景：
    - 电子秤重量显示（如 140.39g 被识别为 14039）
    - 恒温水浴锅温度显示（如 90.5℃ 被识别为 905）
    - 其他七段数码管显示的仪器
    """
    
    # 仪器类型对应的小数点修正规则
    # 格式: {仪器类型: {字段名: {rule, range, decimals}}}
    DECIMAL_RULES = {
        # 电子秤：通常显示小数点后2位
        "electronic_scale": {
            "weight": {
                "rule": "from_right_2",  # 从右边数第2位插入小数点
                "range": [0, 10000],     # 合理重量范围(g)
                "decimals": 2,
                "unit": "g",
            },
            "default": {
                "rule": "from_right_2",
                "range": [0, 10000],
                "decimals": 2,
            },
        },
        # 恒温水浴锅：温度通常显示1位小数
        "water_bath": {
            "temperature": {
                "rule": "from_right_1",  # 从右边数第1位插入小数点
                "range": [0, 100],       # 合理温度范围(℃)
                "decimals": 1,
                "unit": "℃",
            },
            "default": {
                "rule": "from_right_1",
                "range": [0, 100],
                "decimals": 1,
            },
        },
        # pH计/酸度计：pH通常显示2位小数，温度1位小数
        "ph_meter": {
            "ph": {
                "rule": "from_right_2",
                "range": [0, 14],        # pH范围
                "decimals": 2,
            },
            "ph_value": {
                "rule": "from_right_2",
                "range": [0, 14],        # pH范围（与ph字段相同）
                "decimals": 2,
            },
            "temperature": {
                "rule": "from_right_1",
                "range": [0, 100],
                "decimals": 1,
                "unit": "℃",
            },
            "mv_value": {
                "rule": "none",          # mV值通常是整数（可为负数）
                "range": [-2000, 2000],
                "decimals": 0,
                "unit": "mV",
            },
            "conductivity": {
                "rule": "none",          # 电导率通常是整数
                "range": [0, 10000],
                "decimals": 0,
                "unit": "μS/cm",
            },
            "default": {
                "rule": "from_right_2",
                "range": [0, 14],
                "decimals": 2,
            },
        },
        # 电导率仪
        "conductivity_meter": {
            "conductivity": {
                "rule": "none",
                "range": [0, 100000],
                "decimals": 0,
                "unit": "μS/cm",
            },
            "temperature": {
                "rule": "from_right_1",
                "range": [0, 100],
                "decimals": 1,
            },
            "default": {
                "rule": "none",
                "range": [0, 100000],
                "decimals": 0,
            },
        },
        # 混调器：转速通常是整数或2位小数
        "mixer_stirrer": {
            "current_rpm": {
                "rule": "smart_rpm",     # 智能判断
                "range": [0, 5000],
                "decimals": 0,
                "unit": "rpm",
            },
            "default": {
                "rule": "smart_rpm",
                "range": [0, 5000],
                "decimals": 0,
            },
        },
        # 粘度计：粘度通常显示1-2位小数
        "viscometer": {
            "viscosity": {
                "rule": "from_right_1",
                "range": [0, 1000],
                "decimals": 1,
                "unit": "mPa·s",
            },
            "speed": {
                "rule": "none",
                "range": [0, 1000],
                "decimals": 0,
            },
            "default": {
                "rule": "from_right_1",
                "range": [0, 1000],
                "decimals": 1,
            },
        },
        # 表面张力仪
        "surface_tensiometer": {
            "surface_tension": {
                "rule": "from_right_3",
                "range": [0, 100],
                "decimals": 3,
                "unit": "mN/m",
            },
            "default": {
                "rule": "from_right_3",
                "range": [0, 100],
                "decimals": 3,
            },
        },
    }
    
    # 七段数码管常见数字混淆映射
    # 七段显示中，由于某些段未被OCR识别（如亮度不够、角度问题），
    # 容易产生以下混淆：
    DIGIT_CONFUSIONS = {
        '1': ['7'],       # 缺少顶部横段 → 7误识为1
        '7': ['1'],       # 顶部横段噪声 → 1误识为7
        '0': ['8'],       # 缺少中间横段
        '8': ['0', '6'],  # 多余段
        '6': ['8', '5'],  # 缺少/多余段
        '5': ['6', '9'],  # 相似形状
        '9': ['5', '3'],  # 相似形状
        '3': ['9'],       # 相似形状
    }

    # 各仪器字段的"常见值"范围（用于判断digit confusion修正后哪个更合理）
    # 这与DECIMAL_RULES中的range不同，range是"物理上可能的"范围，
    # 而这里是"实验室日常最常见的"范围
    COMMON_VALUE_RANGES = {
        "ph_meter": {
            "ph": [3.0, 11.0],        # 实验室最常见pH范围
            "ph_value": [3.0, 11.0],
            "temperature": [15.0, 40.0],
        },
    }

    # 仪器类型别名映射
    INSTRUMENT_ALIASES = {
        "scale": "electronic_scale",
        "balance": "electronic_scale",
        "电子秤": "electronic_scale",
        "天平": "electronic_scale",
        "华志": "electronic_scale",
        "TP series": "electronic_scale",
        
        "water_bath": "water_bath",
        "水浴锅": "water_bath",
        "恒温水浴": "water_bath",
        "KXY": "water_bath",
        
        "ph_meter": "ph_meter",
        "ph meter": "ph_meter",
        "pH计": "ph_meter",
        "酸度计": "ph_meter",
        "酸度仪": "ph_meter",
        
        "conductivity_meter": "conductivity_meter",
        "电导率仪": "conductivity_meter",
        "电导仪": "conductivity_meter",
        
        "mixer": "mixer_stirrer",
        "stirrer": "mixer_stirrer",
        "混调器": "mixer_stirrer",
        "搅拌器": "mixer_stirrer",
        "LABHTO": "mixer_stirrer",
        "LABHTD": "mixer_stirrer",
        "力辰": "mixer_stirrer",
        "LICHEN": "mixer_stirrer",
        
        "viscometer": "viscometer",
        "粘度计": "viscometer",
        "VISCOMETER": "viscometer",
        
        "tensiometer": "surface_tensiometer",
        "张力仪": "surface_tensiometer",
        "表面张力": "surface_tensiometer",
    }
    
    def __init__(self):
        pass
    
    def normalize_instrument_type(self, instrument_type: str) -> str:
        """标准化仪器类型名称"""
        if not instrument_type:
            return "unknown"
        
        # 直接匹配
        if instrument_type in self.DECIMAL_RULES:
            return instrument_type
        
        # 别名匹配
        for alias, standard_type in self.INSTRUMENT_ALIASES.items():
            if alias.lower() in instrument_type.lower():
                return standard_type
        
        return instrument_type
    
    def fix_decimal_point(
        self,
        value: Union[int, float, str],
        instrument_type: str,
        field_name: str = "default",
    ) -> Tuple[float, bool, str]:
        """
        修正七段数码管读数的小数点
        
        Args:
            value: 原始值（可能缺少小数点）
            instrument_type: 仪器类型
            field_name: 字段名称
            
        Returns:
            (修正后的值, 是否进行了修正, 修正说明)
        """
        # 转换为数字
        if isinstance(value, str):
            # 移除单位和空格
            clean_value = re.sub(r'[^\d.-]', '', value)
            if not clean_value:
                return 0.0, False, "无法解析数值"
            try:
                num_value = float(clean_value)
            except ValueError:
                return 0.0, False, "无法解析数值"
        else:
            num_value = float(value)
        
        # 如果已经有小数点（原始值是浮点数且不是整数），可能不需要修正
        original_str = str(value)
        has_decimal = '.' in original_str
        
        # 标准化仪器类型
        norm_type = self.normalize_instrument_type(instrument_type)
        
        # 获取修正规则
        rules = self.DECIMAL_RULES.get(norm_type, {})
        rule_config = rules.get(field_name, rules.get("default", {}))
        
        if not rule_config:
            return num_value, False, "无适用规则"
        
        rule = rule_config.get("rule", "none")
        value_range = rule_config.get("range", [0, float('inf')])
        expected_decimals = rule_config.get("decimals", 0)
        
        # 如果原始值已有小数点且在合理范围内，不修正
        if has_decimal and value_range[0] <= num_value <= value_range[1]:
            return num_value, False, "原值已在合理范围"
        
        # 应用修正规则
        corrected_value, was_corrected, reason = self._apply_rule(
            num_value, rule, value_range, expected_decimals
        )
        
        return corrected_value, was_corrected, reason
    
    def _apply_rule(
        self,
        value: float,
        rule: str,
        value_range: List[float],
        expected_decimals: int,
    ) -> Tuple[float, bool, str]:
        """应用小数点修正规则"""
        
        if rule == "none" or not rule:
            return value, False, "规则为none"
        
        # ★★★ 关键修改：对于可能缺少小数点的情况，即使在范围内也要检查 ★★★
        # 如果是整数且位数>=4，很可能缺少小数点
        is_likely_missing_decimal = (
            value == int(value) and  # 是整数
            value >= 1000 and        # 位数>=4
            expected_decimals > 0    # 期望有小数位
        )
        
        # 如果值已在合理范围内，且不太可能缺少小数点，则不修正
        if value_range[0] <= value <= value_range[1] and not is_likely_missing_decimal:
            return value, False, "值已在合理范围"
        
        if rule.startswith("from_right_"):
            # 从右边数第n位插入小数点
            try:
                n = int(rule.replace("from_right_", ""))
                divisor = 10 ** n
                corrected = value / divisor
                
                # 检查修正后是否在合理范围
                if value_range[0] <= corrected <= value_range[1]:
                    return corrected, True, f"从右边第{n}位插入小数点: {value} -> {corrected}"
                else:
                    # 尝试其他位置
                    for try_n in range(1, 4):
                        try_corrected = value / (10 ** try_n)
                        if value_range[0] <= try_corrected <= value_range[1]:
                            return try_corrected, True, f"智能修正: {value} -> {try_corrected}"
                    
                    return value, False, f"修正后({corrected})不在合理范围"
            except (ValueError, ZeroDivisionError):
                return value, False, "规则解析失败"
        
        elif rule == "smart_rpm":
            # 智能转速修正
            # 转速通常是整数，但如果值过大，可能缺少小数点
            if value > value_range[1]:
                # 尝试插入小数点
                for n in range(1, 4):
                    corrected = value / (10 ** n)
                    if value_range[0] <= corrected <= value_range[1]:
                        return corrected, True, f"智能RPM修正: {value} -> {corrected}"
            return value, False, "RPM值合理或无法修正"
        
        elif rule == "smart_detect":
            # 智能检测：尝试各种小数点位置
            for n in range(1, 4):
                corrected = value / (10 ** n)
                if value_range[0] <= corrected <= value_range[1]:
                    return corrected, True, f"智能检测: {value} -> {corrected}"
            return value, False, "无法智能修正"
        
        return value, False, f"未知规则: {rule}"
    
    def fix_all_readings(
        self,
        readings: Dict,
        instrument_type: str,
    ) -> Dict:
        """
        修正所有读数中的小数点
        
        Args:
            readings: 原始读数字典
            instrument_type: 仪器类型
            
        Returns:
            修正后的读数字典
        """
        corrected_readings = {}
        
        for key, value in readings.items():
            if value is None or value == "N/A" or value == "":
                corrected_readings[key] = value
                continue
            
            # 跳过列表类型（如 all_numbers）
            if isinstance(value, list):
                corrected_readings[key] = value
                continue
            
            # 跳过非数值字段
            if isinstance(value, str) and not re.search(r'\d', value):
                corrected_readings[key] = value
                continue
            
            # 尝试修正
            try:
                corrected_value, was_corrected, reason = self.fix_decimal_point(
                    value, instrument_type, key
                )
                
                if was_corrected:
                    logger.info(f"小数点修正 [{key}]: {value} -> {corrected_value} ({reason})")
                    corrected_readings[key] = corrected_value
                else:
                    corrected_readings[key] = value if not isinstance(value, str) else self._try_parse_number(value)
            except Exception as e:
                logger.warning(f"修正 {key} 时出错: {e}")
                corrected_readings[key] = value
        
        return corrected_readings
    
    def _try_parse_number(self, value: str) -> Union[float, str]:
        """尝试将字符串解析为数字"""
        try:
            clean = re.sub(r'[^\d.-]', '', value)
            if clean:
                return float(clean)
        except:
            pass
        return value

    def fix_digit_confusion(
        self,
        value: float,
        instrument_type: str,
        field_name: str,
    ) -> Tuple[float, bool, str]:
        """
        修正七段数码管的数字混淆

        七段数码管中，由于某些段未被OCR正确识别，
        可能导致数字被误读（如7被读成1，0被读成8等）。

        本方法通过检查数值是否在"常见范围"外来判断是否存在数字混淆，
        然后尝试用混淆映射替换各个数位，选择落入常见范围的结果。

        Args:
            value: 原始数值
            instrument_type: 仪器类型
            field_name: 字段名

        Returns:
            (修正后的值, 是否修正, 修正说明)
        """
        norm_type = self.normalize_instrument_type(instrument_type)
        common_ranges = self.COMMON_VALUE_RANGES.get(norm_type, {})

        # 查找该字段的常见范围
        common_range = common_ranges.get(field_name)
        if not common_range:
            return value, False, "无常见范围定义"

        # 如果值已经在常见范围内，不修正
        if common_range[0] <= value <= common_range[1]:
            return value, False, "值在常见范围内"

        # 获取物理上可能的范围（用于验证替换结果）
        rules = self.DECIMAL_RULES.get(norm_type, {})
        rule_config = rules.get(field_name, rules.get("default", {}))
        valid_range = rule_config.get("range", [0, float('inf')]) if rule_config else [0, float('inf')]

        # 将数值转换为字符串，逐位尝试替换混淆数字
        # 保留原始精度
        if value == int(value):
            value_str = str(int(value))
        else:
            value_str = f"{value:.2f}".rstrip('0').rstrip('.')
            # 确保至少保留到已有精度
            if '.' in str(value):
                decimals = len(str(value).split('.')[1])
                value_str = f"{value:.{decimals}f}"

        best_value = value
        best_distance = float('inf')
        best_reason = ""

        # 计算常见范围的中心值
        range_center = (common_range[0] + common_range[1]) / 2

        for i, char in enumerate(value_str):
            if char == '.' or char == '-':
                continue
            if char not in self.DIGIT_CONFUSIONS:
                continue
            for alt_digit in self.DIGIT_CONFUSIONS[char]:
                alt_str = value_str[:i] + alt_digit + value_str[i+1:]
                try:
                    alt_value = float(alt_str)
                except ValueError:
                    continue

                # 检查替换后的值是否在常见范围内且物理上合理
                if (common_range[0] <= alt_value <= common_range[1] and
                        valid_range[0] <= alt_value <= valid_range[1]):
                    distance = abs(alt_value - range_center)
                    if distance < best_distance:
                        best_value = alt_value
                        best_distance = distance
                        best_reason = (
                            f"七段数码管数字混淆修正: {value} -> {alt_value} "
                            f"('{char}'->'{alt_digit}' 在位置{i})"
                        )

        if best_value != value:
            return best_value, True, best_reason

        return value, False, "无法通过数字混淆修正"

    def fix_all_readings_with_confusion(
        self,
        readings: Dict,
        instrument_type: str,
    ) -> Dict:
        """
        在小数点修正之后，额外进行数字混淆修正

        Args:
            readings: 读数字典（已完成小数点修正）
            instrument_type: 仪器类型

        Returns:
            修正后的读数字典
        """
        corrected = {}
        for key, value in readings.items():
            if isinstance(value, (int, float)):
                corrected_val, was_corrected, reason = self.fix_digit_confusion(
                    float(value), instrument_type, key
                )
                if was_corrected:
                    logger.info(f"数字混淆修正 [{key}]: {value} -> {corrected_val} ({reason})")
                    corrected[key] = corrected_val
                else:
                    corrected[key] = value
            else:
                corrected[key] = value
        return corrected

    def detect_instrument_from_ocr(self, ocr_results: List[Dict]) -> str:
        """
        从OCR结果中检测仪器类型
        
        Args:
            ocr_results: OCR识别结果列表
            
        Returns:
            检测到的仪器类型
        """
        all_text = " ".join(r.get("text", "") for r in ocr_results).upper()
        
        # ★★★ 先检查单位，这是最可靠的识别方式 ★★★
        # 表面张力仪：mN/m 单位是独特标识
        if re.search(r'MN\s*/\s*M|MN/M', all_text, re.I):
            logger.debug("检测到仪器类型: surface_tensiometer (单位: mN/m)")
            return "surface_tensiometer"
        
        # pH计：pH值通常在0-14之间，配合温度显示
        # 检查是否有pH相关标识
        if re.search(r'\bPH\b|酸度计|酸度仪', all_text, re.I):
            logger.debug("检测到仪器类型: ph_meter (关键词)")
            return "ph_meter"
        
        # 电导率仪：μS/cm 或 mS/cm 单位
        if re.search(r'[μU]S\s*/\s*CM|MS\s*/\s*CM|电导率', all_text, re.I):
            logger.debug("检测到仪器类型: conductivity_meter (单位)")
            return "conductivity_meter"
        
        # 粘度计：mPa·s 单位，且必须有VISCOMETER关键词
        if "VISCOMETER" in all_text or "粘度计" in all_text:
            logger.debug("检测到仪器类型: viscometer (关键词)")
            return "viscometer"
        
        # 优先级检测（按可靠性排序）
        detection_patterns = [
            # 表面张力仪 - 高优先级
            (["SURFACE TENSION", "TENSIOMETER", "表面张力", "界面张力"], "surface_tensiometer"),
            # pH计/酸度计
            (["PH METER", "酸度计", "PH计"], "ph_meter"),
            # 电导率仪
            (["CONDUCTIVITY", "电导率仪", "电导仪"], "conductivity_meter"),
            # 电子秤
            (["华志", "HUAZHI", "TP SERIES", "TP-", "BALANCE"], "electronic_scale"),
            # 水浴锅
            (["KXY", "恒温水浴", "水浴锅", "WATER BATH"], "water_bath"),
            # 混调器/搅拌器
            (["混调器", "高速", "低速"], "mixer_stirrer"),
            (["LABHTO", "LABHTD", "青岛恒泰达"], "mixer_stirrer"),
            (["力辰", "LICHEN", "搅拌器", "STIRRER"], "mixer_stirrer"),
            # 粘度计 - 必须有明确的VISCOMETER标识，放在最后
            (["VISCOMETER", "粘度计"], "viscometer"),
        ]
        
        for keywords, inst_type in detection_patterns:
            for kw in keywords:
                if kw.upper() in all_text:
                    # 特殊处理：避免误匹配
                    if inst_type == "viscometer":
                        # 粘度计需要明确的VISCOMETER标识
                        if "VISCOMETER" not in all_text and "粘度计" not in all_text:
                            continue
                    logger.debug(f"检测到仪器类型: {inst_type} (关键词: {kw})")
                    return inst_type
        
        # 基于单位的后备检测
        if re.search(r'\d+\.?\d*\s*g\b', all_text, re.I):
            return "electronic_scale"
        if re.search(r'\d+\.?\d*\s*[°℃]', all_text):
            return "water_bath"
        if re.search(r'\d+\.?\d*\s*RPM|转速|转/分', all_text, re.I):
            return "mixer_stirrer"
        
        # ★★★ 智能推断：基于数值特征 ★★★
        # 提取所有数字
        numbers = re.findall(r'\d+\.?\d*', all_text)
        if numbers:
            nums = [float(n) for n in numbers if n]
            # pH计特征：有一个6-8左右的数（pH值）和一个20-30左右的数（温度）
            has_ph_like = any(6.0 <= n <= 8.5 for n in nums)
            has_temp_like = any(20 <= n <= 35 for n in nums)
            if has_ph_like and has_temp_like:
                logger.debug("检测到仪器类型: ph_meter (数值特征推断)")
                return "ph_meter"
        
        return "unknown"


# =====================================================
# 仪器配置加载
# =====================================================
def load_instrument_schemas() -> Dict[str, InstrumentSchema]:
    """加载仪器模式配置"""
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
                fields=config.get("fields", {}),
                keywords=config.get("keywords", []),
                required_fields=required,
                has_tabs=config.get("has_tabs", False),
                tabs=config.get("tabs", []),
            )
        return schemas
    except ImportError:
        logger.warning("无法加载config.py，使用默认配置")
        return {}


INSTRUMENT_SCHEMAS = load_instrument_schemas()

# 全局小数点修正器实例
decimal_fixer = SegmentDisplayDecimalFixer()


# =====================================================
# LLM基类
# =====================================================
class LLMBase(ABC):
    """LLM基类"""
    
    def __init__(self):
        self.decimal_fixer = SegmentDisplayDecimalFixer()
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """生成文本"""
        pass
    
    def validate_ocr_results(
        self,
        ocr_results: List[Dict],
        instrument_type: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        验证OCR结果是否足够
        
        Returns:
            (是否足够, 原因/建议)
        """
        if not ocr_results:
            return False, "OCR未识别到任何文字"
        
        # 检查平均置信度
        avg_conf = sum(r.get("confidence", 0) for r in ocr_results) / len(ocr_results)
        if avg_conf < 0.5:
            return False, f"OCR置信度过低 ({avg_conf:.2f})，建议增强对比度重试"
        
        # 检查是否有数字
        has_numbers = any(re.search(r'\d', r.get("text", "")) for r in ocr_results)
        if not has_numbers:
            return False, "未识别到任何数字，可能需要旋转图片或调整预处理参数"
        
        # 如果指定了仪器类型，检查必要字段
        if instrument_type and instrument_type in INSTRUMENT_SCHEMAS:
            schema = INSTRUMENT_SCHEMAS[instrument_type]
            all_text = " ".join(r.get("text", "") for r in ocr_results)
            
            # 检查关键词匹配
            keyword_matches = sum(1 for kw in schema.keywords if kw in all_text)
            if keyword_matches == 0:
                return False, f"未匹配到{instrument_type}的关键词，可能识别不准确"
        
        return True, "OCR结果足够"
    
    def parse_instrument_reading(
        self,
        ocr_results: List[Dict],
        instrument_type: Optional[str] = None,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> ParseResult:
        """
        解析仪器读数（带验证和小数点修正）
        
        Args:
            ocr_results: OCR识别结果列表
            instrument_type: 仪器类型
            attempt: 当前尝试次数
            max_attempts: 最大尝试次数
            
        Returns:
            ParseResult对象
        """
        # 首先验证OCR结果
        is_valid, reason = self.validate_ocr_results(ocr_results, instrument_type)
        
        if not is_valid:
            if attempt < max_attempts:
                return ParseResult(
                    status=ValidationResult.NEED_RETRY,
                    message=reason,
                    retry_suggestion="increase_contrast" if "对比度" in reason else "try_rotation",
                )
            else:
                return ParseResult(
                    status=ValidationResult.FAILED,
                    message=f"经过{max_attempts}次尝试仍无法识别: {reason}",
                )
        
        # 构建OCR文本
        ocr_text = self._format_ocr_for_prompt(ocr_results)
        
        # 自动识别仪器类型（如果未指定）
        if instrument_type is None:
            instrument_type = self._identify_instrument_type(ocr_results)
        
        schema = INSTRUMENT_SCHEMAS.get(instrument_type)
        
        # 构建prompt并调用LLM
        prompt = self._build_parsing_prompt(ocr_text, schema, instrument_type)
        response = self.generate(prompt)
        
        # 解析JSON
        result = self._extract_json_from_response(response)
        
        # 获取读数
        readings = result.get("readings", {})
        
        # ★★★ 关键修改：应用小数点修正 ★★★
        if readings and isinstance(readings, dict):
            # 检测仪器类型（如果还是unknown）
            detected_type = self.decimal_fixer.detect_instrument_from_ocr(ocr_results)
            if instrument_type == "unknown" and detected_type != "unknown":
                instrument_type = detected_type
                logger.info(f"从OCR结果检测到仪器类型: {instrument_type}")
            
            # 应用小数点修正
            corrected_readings = self.decimal_fixer.fix_all_readings(readings, instrument_type)
            # 应用七段数码管数字混淆修正（如 1↔7 混淆）
            corrected_readings = self.decimal_fixer.fix_all_readings_with_confusion(
                corrected_readings, instrument_type
            )
            result["readings"] = corrected_readings
            readings = corrected_readings

        # 验证解析结果
        confidence = self._calculate_confidence(readings, schema, ocr_results)
        
        # 检查是否有足够的数据
        if not readings or (isinstance(readings, dict) and len(readings) == 0):
            if attempt < max_attempts:
                return ParseResult(
                    status=ValidationResult.NEED_RETRY,
                    message="未能提取到有效读数",
                    retry_suggestion="aggressive_preprocessing",
                )
            else:
                return ParseResult(
                    status=ValidationResult.FAILED,
                    message="无法从OCR结果中提取有效数据",
                    data={"raw_ocr": [r["text"] for r in ocr_results]},
                    raw_ocr_texts=[r["text"] for r in ocr_results],
                )
        
        # 检查必要字段
        missing_fields = []
        if schema:
            for field in schema.required_fields:
                if field not in readings or readings[field] in [None, "", "N/A"]:
                    missing_fields.append(field)
        
        if missing_fields and attempt < max_attempts:
            return ParseResult(
                status=ValidationResult.NEED_RETRY,
                message=f"缺少必要字段: {missing_fields}",
                missing_fields=missing_fields,
                retry_suggestion="try_rotation",
            )
        
        # 成功
        result["instrument_type"] = instrument_type
        result["raw_ocr"] = [r["text"] for r in ocr_results]
        result["confidence"] = confidence
        
        return ParseResult(
            status=ValidationResult.SUCCESS,
            data=result,
            confidence=confidence,
            message="识别成功",
            raw_ocr_texts=[r["text"] for r in ocr_results],
        )
    
    def _format_ocr_for_prompt(self, ocr_results: List[Dict]) -> str:
        """格式化OCR结果"""
        lines = []
        for r in ocr_results:
            lines.append(f"- {r['text']} (置信度: {r.get('confidence', 0):.2f})")
        return "\n".join(lines)
    
    def _identify_instrument_type(self, ocr_results: List[Dict]) -> str:
        """
        自动识别仪器类型 - 完全重写版
        
        识别优先级：
        1. 明确的单位标识（最可靠）
        2. 明确的仪器名称关键词
        3. 数值特征推断
        4. 默认unknown
        
        重要原则：粘度计必须有明确的VISCOMETER标识！
        """
        all_text = " ".join(r.get("text", "") for r in ocr_results)
        all_text_upper = all_text.upper()
        
        # ========== 第一优先级：明确的单位标识 ==========
        # 表面张力仪：mN/m 是独特标识
        if re.search(r'MN\s*/\s*M|MN/M', all_text_upper):
            logger.info("仪器识别: surface_tensiometer (单位mN/m)")
            return "surface_tensiometer"
        
        # 电导率：μS/cm 或 mS/cm
        if re.search(r'[μuU]S\s*/\s*CM|MS\s*/\s*CM', all_text_upper):
            logger.info("仪器识别: conductivity_meter (单位μS/cm)")
            return "conductivity_meter"
        
        # ========== 第二优先级：明确的仪器名称 ==========
        # 粘度计：必须有明确的 VISCOMETER 标识
        if "VISCOMETER" in all_text_upper:
            logger.info("仪器识别: viscometer (关键词VISCOMETER)")
            return "viscometer"
        
        # pH计
        if re.search(r'\bPH\b', all_text_upper) or "酸度计" in all_text or "酸度仪" in all_text:
            logger.info("仪器识别: ph_meter (关键词pH)")
            return "ph_meter"
        
        # 电子秤
        if any(kw in all_text_upper for kw in ["华志", "HUAZHI", "TP SERIES", "BALANCE"]):
            logger.info("仪器识别: electronic_scale (品牌关键词)")
            return "electronic_scale"
        if re.search(r'\d+\.?\d*\s*[Gg]\b', all_text):  # 带g单位
            logger.info("仪器识别: electronic_scale (单位g)")
            return "electronic_scale"
        
        # 水浴锅
        if any(kw in all_text for kw in ["KXY", "恒温水浴", "水浴锅"]):
            logger.info("仪器识别: water_bath (关键词)")
            return "water_bath"
        
        # 混调器
        if any(kw in all_text for kw in ["混调器", "高速", "低速"]):
            if "LABHTO" in all_text_upper or "LABHTD" in all_text_upper or "青岛恒泰达" in all_text:
                logger.info("仪器识别: mixer_stirrer (混调器)")
                return "mixer_stirrer"
        
        # 搅拌器（力辰等）
        if any(kw in all_text for kw in ["力辰", "LICHEN", "搅拌器"]):
            logger.info("仪器识别: mixer_stirrer (搅拌器)")
            return "mixer_stirrer"
        if re.search(r'\d+\s*RPM', all_text_upper):
            logger.info("仪器识别: mixer_stirrer (单位RPM)")
            return "mixer_stirrer"
        
        # 表面张力仪（关键词）
        if any(kw in all_text for kw in ["表面张力", "界面张力", "TENSIOMETER", "SURFACE TENSION"]):
            logger.info("仪器识别: surface_tensiometer (关键词)")
            return "surface_tensiometer"
        
        # 水质检测仪
        if any(kw in all_text for kw in ["检测结果", "吸光度", "透光度", "空白值", "检测值"]):
            logger.info("仪器识别: water_quality_meter (关键词)")
            return "water_quality_meter"
        
        # ========== 第三优先级：数值特征推断 ==========
        numbers = []
        for r in ocr_results:
            matches = re.findall(r'(\d+\.?\d*)', r.get("text", ""))
            for m in matches:
                try:
                    numbers.append(float(m))
                except:
                    pass
        
        if numbers:
            # pH计特征推断（考虑小数点可能缺失）
            # 正常pH值：6-9 之间
            # 缺少小数点：60-90（如7.49变成749）或 6-9xx（如749）
            
            has_ph_like = False
            has_temp_like = False
            has_conductivity_like = False
            
            for n in numbers:
                # 正常pH值范围
                if 6.0 <= n <= 9.0:
                    has_ph_like = True
                # pH值缺少小数点（如7.49变成749，或14.9变成149）
                elif 60 <= n <= 149:
                    # 可能是缺少小数点的pH值(6.0-14.9)
                    has_ph_like = True
                # 温度范围（正常或缺少小数点）
                elif 20 <= n <= 40:
                    has_temp_like = True
                elif 200 <= n <= 400:
                    # 可能是缺少小数点的温度（如25.0变成250）
                    has_temp_like = True
                # 电导率（通常是3-4位整数）
                elif 100 <= n <= 9999:
                    has_conductivity_like = True
            
            # pH计通常同时显示：pH值 + 温度，或 pH值 + 温度 + 电导率
            if has_ph_like and has_temp_like:
                logger.info(f"仪器识别: ph_meter (数值特征: {numbers})")
                return "ph_meter"
            
            # 如果有电导率特征配合温度，也可能是pH计或电导率仪
            if has_conductivity_like and has_temp_like:
                logger.info(f"仪器识别: ph_meter (电导率+温度特征: {numbers})")
                return "ph_meter"
        
        # ========== 默认 ==========
        logger.info(f"仪器识别: unknown (无法匹配, OCR: {all_text[:100]})")
        return "unknown"
    
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
                + (f" [必需]" if v.get('required') else "")
                for k, v in schema.fields.items()
            )
            
            tabs_info = ""
            if schema.has_tabs:
                tabs_info = f"\n注意：该仪器有多个选项卡/模式: {schema.tabs}，请识别当前显示的模式。"
            
            prompt = f"""你是一个专业的仪器数据解析助手。请根据以下OCR识别结果，提取仪器读数并以JSON格式输出。

仪器类型: {instrument_type} ({schema.fields.get('name', instrument_type) if isinstance(schema.fields, dict) else instrument_type})
{tabs_info}

需要提取的字段:
{fields_desc}

OCR识别结果:
{ocr_text}

【重要】七段数码管小数点处理说明：
- 七段数码管的小数点经常无法被OCR正确识别
- 如果识别到的数字位数较多但没有小数点，请根据仪器类型判断是否需要添加小数点
- 电子秤：通常显示2位小数（如 14039 应该是 140.39）
- 温度计/水浴锅：通常显示1位小数（如 905 应该是 90.5）
- 请根据数值的合理范围来判断小数点位置

请严格输出以下JSON格式，只输出JSON，不要有其他内容:
{{
    "readings": {{
        "字段名": 数值或字符串,
        ...
    }},
    "confidence": 0.0到1.0之间的数值
}}

如果某个字段无法从OCR结果中提取，请设为null或"N/A"。
对于数值字段，请只提取数字部分（可以保留小数点）。
"""
        else:
            prompt = f"""你是一个专业的仪器数据解析助手。请根据以下OCR识别结果，智能识别并提取仪器读数。

OCR识别结果:
{ocr_text}

【重要】七段数码管小数点处理说明：
- 七段数码管的小数点经常无法被OCR正确识别
- 如果识别到的数字位数较多但没有小数点，请根据上下文判断是否需要添加小数点
- 电子秤：通常显示2位小数（如 14039 应该是 140.39）
- 温度显示：通常显示1位小数（如 905 应该是 90.5）

请分析这些文字，提取所有有意义的读数和参数，以JSON格式输出:
{{
    "readings": {{
        "参数名": 数值或字符串,
        ...
    }},
    "instrument_type": "推测的仪器类型",
    "confidence": 0.0到1.0之间的置信度
}}

只输出JSON，不要有其他内容。
"""
        return prompt
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """从LLM响应中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 提取```json```块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 提取{}块
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"无法从响应中提取JSON: {response[:200]}...")
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
        
        # 基础分：OCR平均置信度
        if ocr_results:
            base_conf = sum(r.get("confidence", 0) for r in ocr_results) / len(ocr_results)
        else:
            base_conf = 0.5
        
        # 字段完整性
        if schema:
            total_fields = len(schema.fields)
            filled_fields = sum(
                1 for k, v in readings.items()
                if v not in [None, "", "N/A", "null"]
            )
            field_ratio = filled_fields / total_fields if total_fields > 0 else 0.5
        else:
            field_ratio = 0.5 if readings else 0.0
        
        # 综合置信度
        confidence = base_conf * 0.4 + field_ratio * 0.6
        
        return min(max(confidence, 0.0), 1.0)


# =====================================================
# 基于规则的解析器
# =====================================================
class RuleBasedParser(LLMBase):
    """基于规则的解析器（不需要LLM）"""
    
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
        # 验证OCR结果
        is_valid, reason = self.validate_ocr_results(ocr_results, instrument_type)
        
        if not is_valid:
            if attempt < max_attempts:
                return ParseResult(
                    status=ValidationResult.NEED_RETRY,
                    message=reason,
                    retry_suggestion="increase_contrast",
                )
            else:
                return ParseResult(
                    status=ValidationResult.FAILED,
                    message=f"经过{max_attempts}次尝试仍无法识别: {reason}",
                )
        
        # 识别仪器类型
        if instrument_type is None:
            instrument_type = self._identify_instrument_type(ocr_results)
            
        # 也尝试从OCR结果检测
        if instrument_type == "unknown":
            instrument_type = self.decimal_fixer.detect_instrument_from_ocr(ocr_results)
        
        schema = INSTRUMENT_SCHEMAS.get(instrument_type)

        # 提取键值对
        all_text = [r["text"] for r in ocr_results]
        raw_readings = self._extract_readings_by_rules(ocr_results, schema)

        # ★★★ pH计特殊处理：使用空间分析提取数值 ★★★
        if instrument_type == "ph_meter":
            ph_readings = self._extract_ph_meter_readings(ocr_results)
            # 将pH专用提取结果合并到raw_readings（pH专用结果优先）
            for key, value in ph_readings.items():
                if key not in raw_readings or raw_readings[key] is None:
                    raw_readings[key] = value
                elif key == "ph_value" and value is not None:
                    # pH值始终用空间分析的结果覆盖
                    raw_readings[key] = value

        # 映射到标准字段
        mapped_readings = self._map_to_schema(raw_readings, schema, instrument_type)

        # ★★★ 应用小数点修正 ★★★
        corrected_readings = self.decimal_fixer.fix_all_readings(mapped_readings, instrument_type)

        # ★★★ 应用七段数码管数字混淆修正（如 1↔7 混淆）★★★
        corrected_readings = self.decimal_fixer.fix_all_readings_with_confusion(
            corrected_readings, instrument_type
        )

        # 计算置信度
        confidence = self._calculate_confidence(corrected_readings, schema, ocr_results)
        
        # 检查结果
        if not corrected_readings and not raw_readings:
            if attempt < max_attempts:
                return ParseResult(
                    status=ValidationResult.NEED_RETRY,
                    message="未能提取到有效读数",
                    retry_suggestion="try_rotation",
                )
            else:
                return ParseResult(
                    status=ValidationResult.FAILED,
                    message="无法提取有效数据",
                    data={"raw_ocr": all_text},
                    raw_ocr_texts=all_text,
                )
        
        result = {
            "instrument_type": instrument_type,
            "readings": corrected_readings,
            "raw_readings": raw_readings,
            "raw_ocr": all_text,
            "confidence": confidence,
        }
        
        return ParseResult(
            status=ValidationResult.SUCCESS,
            data=result,
            confidence=confidence,
            message="识别成功",
            raw_ocr_texts=all_text,
        )
    
    def _extract_readings_by_rules(
        self,
        ocr_results: List[Dict],
        schema: Optional[InstrumentSchema],
    ) -> Dict:
        """基于规则提取读数"""
        readings = {}
        all_text = " ".join(r["text"] for r in ocr_results)
        
        # 特殊模式匹配
        patterns = {
            # 数值+单位模式
            "weight_g": (r'(\d+\.?\d*)\s*g\b', "weight", "g"),
            "weight_kg": (r'(\d+\.?\d*)\s*kg\b', "weight", "kg"),
            "temp_c": (r'(\d+\.?\d*)\s*[°℃]C?', "temperature", "℃"),
            "humidity": (r'(\d+\.?\d*)\s*%\s*(?:RH)?', "humidity", "%RH"),
            "tension": (r'(\d+\.?\d*)\s*mN/m', "surface_tension", "mN/m"),
            "viscosity": (r'(\d+\.?\d*)\s*mPa', "viscosity", "mPa·s"),
            "rpm": (r'(\d+)\s*(?:转|rpm|RPM)', "current_rpm", "转"),
            "density": (r'(\d+\.?\d*)\s*g/cm', "density", "g/cm³"),
        }
        
        for key, (pattern, field_name, unit) in patterns.items():
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    readings[field_name] = value
                    readings[f"{field_name}_unit"] = unit
                except ValueError:
                    pass
        
        # 键值对提取
        for r in ocr_results:
            text = r["text"]
            # 检查分隔符
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
        for i, r in enumerate(ocr_results[:-1]):
            text = r["text"].rstrip(":：=")
            next_text = ocr_results[i + 1]["text"]
            
            # 如果当前是标签（中文或英文词），下一个是数值
            if self._is_label(text) and self._contains_number(next_text):
                value = self._parse_value(next_text)
                if value is not None:
                    readings[text] = value
        
        # ★★★ 提取所有独立的数字（可能是读数）★★★
        extracted_numbers = []
        for r in ocr_results:
            text = r["text"].strip()
            # 匹配数字（包括小数）
            if re.match(r'^-?\d+\.?\d*$', text):
                try:
                    num = float(text)
                    extracted_numbers.append(num)
                except:
                    pass
        
        # 如果有提取到数字，按大小排序后存储
        if extracted_numbers:
            # 最大的数字作为主显示值
            if "main_display" not in readings:
                readings["main_display"] = max(extracted_numbers)
            # 所有数字都保存
            readings["all_numbers"] = extracted_numbers
        
        return readings
    
    def _extract_ph_meter_readings(self, ocr_results: List[Dict]) -> Dict:
        """
        pH计专用数值提取方法

        pH计显示屏通常布局为：
        - 大字体显示pH值（屏幕上半部分，字体最大）
        - 小字体显示温度（屏幕下半部分左侧）
        - 小字体显示mV值或电导率（屏幕下半部分右侧）

        本方法使用OCR结果的位置信息（bounding box）来判断各数值的含义。
        """
        readings = {}

        # 收集所有含数字的OCR结果及其位置
        number_items = []
        for r in ocr_results:
            text = r.get("text", "").strip()
            box = r.get("box", [[0, 0], [100, 0], [100, 30], [0, 30]])
            match = re.search(r'(-?\d+\.?\d*)', text)
            if not match:
                continue
            try:
                num = float(match.group(1))
            except ValueError:
                continue

            y_positions = [p[1] for p in box]
            x_positions = [p[0] for p in box]
            center_y = sum(y_positions) / len(y_positions)
            center_x = sum(x_positions) / len(x_positions)
            height = max(y_positions) - min(y_positions)

            number_items.append({
                "value": num,
                "text": text,
                "center_y": center_y,
                "center_x": center_x,
                "height": height,
            })

        if not number_items:
            return readings

        # 按字体高度排序（降序），最大的通常是pH主显示值
        number_items.sort(key=lambda x: x["height"], reverse=True)

        # 候选pH值：选择字体最大的数字
        ph_assigned = False
        for item in number_items:
            val = item["value"]
            # pH值在0-14范围（或缺少小数点：0-1400）
            if 0 <= val <= 14:
                readings["ph_value"] = val
                ph_assigned = True
                break
            elif 0 < val <= 1400 and val == int(val):
                # 可能缺少小数点，如 749 -> 7.49
                corrected = val / 100.0
                if 0 <= corrected <= 14:
                    readings["ph_value"] = val  # 先存原始值，后续小数点修正器处理
                    ph_assigned = True
                    break

        # 如果最大字体不在pH范围内，仍然把它作为pH候选
        if not ph_assigned and number_items:
            readings["ph_value"] = number_items[0]["value"]
            ph_assigned = True

        # 剩余数字按位置分配：温度和mV值
        remaining = [item for item in number_items
                     if item["value"] != readings.get("ph_value")]

        for item in remaining:
            val = item["value"]
            text = item["text"]

            # 判断是否是温度
            # 温度：15-45 正常范围，或 150-450 缺少小数点
            has_temp_unit = bool(re.search(r'[°℃]', text))
            if has_temp_unit or (15 <= val <= 45):
                readings.setdefault("temperature", val)
                continue
            if 150 <= val <= 450 and val == int(val):
                readings.setdefault("temperature", val)  # 小数点修正器处理
                continue

            # 判断是否是mV值
            has_mv_unit = bool(re.search(r'mV|mv', text, re.I))
            if has_mv_unit or (abs(val) > 50 and "temperature" in readings):
                readings.setdefault("mv_value", val)
                continue

            # 其他数值：可能是电导率
            if val > 100:
                readings.setdefault("mv_value", val)

        return readings

    def _map_to_schema(
        self,
        raw_readings: Dict,
        schema: Optional[InstrumentSchema],
        instrument_type: str = "unknown",
    ) -> Dict:
        """映射到标准字段"""
        if not schema:
            return raw_readings

        mapped = {}
        for field_name, field_info in schema.fields.items():
            chinese_name = field_info.get("chinese", "")

            # 直接匹配字段名
            if field_name in raw_readings:
                mapped[field_name] = raw_readings[field_name]
                continue

            # 匹配中文名
            for key, value in raw_readings.items():
                if chinese_name and (chinese_name in key or key in chinese_name):
                    mapped[field_name] = value
                    break

            # 特殊匹配规则
            if field_name not in mapped:
                if field_name == "weight" and "重量" not in str(raw_readings):
                    # 电子秤可能直接显示数字
                    for key, value in raw_readings.items():
                        if isinstance(value, (int, float)) and "g" in str(raw_readings.get(f"{key}_unit", "")):
                            mapped[field_name] = value
                            break
                    # 使用main_display作为重量
                    if field_name not in mapped and "main_display" in raw_readings:
                        mapped[field_name] = raw_readings["main_display"]

        # pH计特殊处理：如果标准字段仍未填充，尝试从all_numbers智能分配
        if instrument_type == "ph_meter":
            all_numbers = raw_readings.get("all_numbers", [])
            if all_numbers and "ph_value" not in mapped:
                # 按数值大小和合理范围分配
                for num in all_numbers:
                    if 0 <= num <= 14 and "ph_value" not in mapped:
                        mapped["ph_value"] = num
                    elif 0 < num <= 1400 and num == int(num) and "ph_value" not in mapped:
                        mapped["ph_value"] = num  # 留给小数点修正
                    elif (15 <= num <= 45 or 150 <= num <= 450) and "temperature" not in mapped:
                        mapped["temperature"] = num
                    elif abs(num) > 50 and "mv_value" not in mapped:
                        mapped["mv_value"] = num

        return mapped
    
    def _is_label(self, text: str) -> bool:
        """判断是否是标签"""
        # 标签通常包含中文或特定英文词，不以数字开头
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        is_english_label = text.lower() in ["speed", "temp", "time", "rpm", "weight"]
        return (has_chinese or is_english_label) and not text[0].isdigit() if text else False
    
    def _contains_number(self, text: str) -> bool:
        """判断是否包含数字"""
        return bool(re.search(r'\d', text))
    
    def _parse_value(self, text: str) -> Union[float, str, None]:
        """解析数值"""
        if not text:
            return None
        
        # 提取数字
        match = re.search(r'([-+]?\d*\.?\d+)', text)
        if match:
            try:
                num = float(match.group(1))
                # 保留单位
                unit = text.replace(match.group(1), "").strip()
                if unit and len(unit) < 10:
                    return f"{num}{unit}"
                return num
            except ValueError:
                pass
        
        return text if text else None


# =====================================================
# Qwen本地模型
# =====================================================
class QwenLLM(LLMBase):
    """Qwen2-7B 模型"""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-7B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        
    def _lazy_init(self):
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"正在加载模型: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True)
            
            model_kwargs = {"trust_remote_code": True, "device_map": self.device}
            
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                model_kwargs["torch_dtype"] = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **model_kwargs)
            
            logger.info("Qwen模型加载成功")
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        self._lazy_init()
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id)
        
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# =====================================================
# Ollama API调用
# =====================================================
class OllamaLLM(LLMBase):
    """通过Ollama API调用本地模型"""
    
    def __init__(
        self,
        model_name: str = "qwen2:7b",
        base_url: str = "http://localhost:11434",
    ):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Ollama API错误: {response.text}")


# =====================================================
# LM Studio API调用
# =====================================================
class LMStudioLLM(LLMBase):
    """通过LM Studio API调用本地模型（OpenAI兼容接口）"""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234",
        model_name: str = "local-model",
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import requests
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=60,
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            raise Exception(f"LM Studio API错误: {response.status_code} - {response.text}")


# =====================================================
# 工厂函数
# =====================================================
def create_llm(model_type: str = "rule", **kwargs) -> LLMBase:
    """LLM工厂函数"""
    if model_type == "rule":
        return RuleBasedParser()
    elif model_type == "qwen":
        return QwenLLM(**kwargs)
    elif model_type == "ollama":
        return OllamaLLM(**kwargs)
    elif model_type == "lmstudio":
        return LMStudioLLM(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# =====================================================
# 便捷函数
# =====================================================
def fix_segment_display_decimal(
    value: Union[int, float, str],
    instrument_type: str,
    field_name: str = "default",
) -> float:
    """
    便捷函数：修正七段数码管的小数点
    
    示例:
        >>> fix_segment_display_decimal(14039, "electronic_scale", "weight")
        140.39
        >>> fix_segment_display_decimal(905, "water_bath", "temperature")
        90.5
    """
    corrected, _, _ = decimal_fixer.fix_decimal_point(value, instrument_type, field_name)
    return corrected
