"""
仪器读数识别系统 - 配置文件
定义支持的仪器类型、OCR参数、重试策略等
"""

# =============================================================================
# OCR 重试配置
# =============================================================================
OCR_RETRY_CONFIG = {
    "max_retries": 3,                    # 最大重试次数
    "confidence_threshold": 0.10,        # 最低置信度阈值（降低以提高通过率）
    "min_text_count": 1,                 # 最少需要识别的文本数量
    "require_numbers": False,            # 是否强制要求有数字
    
    # 不同重试级别的预处理参数
    "retry_presets": [
        {
            "name": "default",
            "enhance_contrast": False,
            "denoise": False,
            "binarize": False,
            "rotation": 0,
        },
        {
            "name": "enhanced",
            "enhance_contrast": True,
            "contrast_factor": 1.8,
            "denoise": True,
            "binarize": False,
            "rotation": 0,
        },
        {
            "name": "aggressive",
            "enhance_contrast": True,
            "contrast_factor": 2.5,
            "denoise": True,
            "binarize": True,
            "binarize_threshold": 127,
            "rotation": 0,
        },
    ],
}

# =============================================================================
# 仪器类型配置
# =============================================================================
INSTRUMENT_CONFIG = {
    # -------------------------------------------------------------------------
    # 水质检测仪 (Water Quality Tester)
    # -------------------------------------------------------------------------
    "water_quality_tester": {
        "name": "水质检测仪",
        "name_en": "Water Quality Tester",
        "keywords": [
            "检测结果", "检测项目", "检测日期", "检测值",
            "吸光度", "透光度", "含量", "空白值",
            "总硬度", "mg/L", "继续检测", "返回", "上传",
            "低量程", "高量程", "氨氮", "COD", "总磷",
            "浊度", "余氯", "总氯", "溶解氧",
        ],
        "fields": {
            "test_item": {
                "chinese": "检测项目",
                "type": "string",
                "required": True,
            },
            "test_date": {
                "chinese": "检测日期",
                "type": "datetime",
                "format": "%Y-%m-%d %H:%M:%S",
            },
            "blank_value": {
                "chinese": "空白值",
                "type": "number",
            },
            "test_value": {
                "chinese": "检测值",
                "type": "number",
            },
            "absorbance": {
                "chinese": "吸光度",
                "type": "number",
            },
            "content": {
                "chinese": "含量",
                "type": "string",
                "unit": "mg/L",
            },
            "transmittance": {
                "chinese": "透光度",
                "type": "string",
                "unit": "%",
            },
        },
        "has_tabs": False,
    },
    
    # -------------------------------------------------------------------------
    # 电子秤 (Electronic Scale)
    # -------------------------------------------------------------------------
    "electronic_scale": {
        "name": "电子秤",
        "name_en": "Electronic Scale",
        "keywords": [
            "g", "kg", "mg", "HUAZHI", "华志", "TP", "series",
            "TARE", "CAL", "MODE", "ON/OFF", "净重", "毛重",
            "METTLER", "TOLEDO", "OHAUS", "SARTORIUS",
        ],
        "fields": {
            "weight": {
                "chinese": "重量",
                "type": "number",
                "required": True,
            },
            "unit": {
                "chinese": "单位",
                "type": "string",
                "default": "g",
            },
            "tare": {
                "chinese": "皮重",
                "type": "number",
            },
            "mode": {
                "chinese": "模式",
                "type": "string",
            },
        },
        "has_tabs": False,
    },
    
    # -------------------------------------------------------------------------
    # 粘度计 (Viscometer)
    # -------------------------------------------------------------------------
    "viscometer": {
        "name": "粘度计",
        "name_en": "Viscometer",
        "keywords": [
            "VISCOMETER", "粘度", "速度", "LABHTD", "恒泰达",
            "mPa", "停止", "清零", "rpm", "转速",
            "3", "6", "100", "200", "300", "600",
            "转子", "spindle", "BROOKFIELD",
        ],
        "fields": {
            "speed": {
                "chinese": "速度",
                "type": "number",
                "unit": "rpm",
            },
            "viscosity": {
                "chinese": "粘度",
                "type": "number",
                "unit": "mPa·s",
            },
            "spindle": {
                "chinese": "转子号",
                "type": "string",
            },
            "torque": {
                "chinese": "扭矩",
                "type": "number",
                "unit": "%",
            },
        },
        "has_tabs": False,
    },
    
    # -------------------------------------------------------------------------
    # 混调器/搅拌器 (Mixer Stirrer)
    # -------------------------------------------------------------------------
    "mixer_stirrer": {
        "name": "混调器/搅拌器",
        "name_en": "Mixer Stirrer",
        "keywords": [
            "LICHEN", "力辰", "rpm", "转", "转速",
            "SET", "Prog", "Push", "ON/OFF", "Ncm",
            "LABHTD", "恒泰达", "混调器", "调速器", "搅拌",
            "高速", "低速", "自动", "手动", "消泡", "无极",
            "启动", "停止", "转速(转)", "时间(S)", "当前转速",
            "IKA", "顶置", "磁力",
        ],
        "fields": {
            "current_rpm": {
                "chinese": "当前转速",
                "type": "number",
                "unit": "rpm",
            },
            "set_rpm": {
                "chinese": "设定转速",
                "type": "number",
                "unit": "rpm",
            },
            "torque": {
                "chinese": "扭矩",
                "type": "number",
                "unit": "Ncm",
            },
            "time": {
                "chinese": "时间",
                "type": "string",
            },
            "mode": {
                "chinese": "模式",
                "type": "string",
            },
            "high_speed_rpm": {
                "chinese": "高速转速",
                "type": "number",
            },
            "high_speed_time": {
                "chinese": "高速时间",
                "type": "number",
                "unit": "s",
            },
            "low_speed_rpm": {
                "chinese": "低速转速",
                "type": "number",
            },
            "low_speed_time": {
                "chinese": "低速时间",
                "type": "number",
                "unit": "s",
            },
        },
        "has_tabs": True,
        "tabs": ["高速", "低速", "自动", "手动", "消泡", "无极"],
    },
    
    # -------------------------------------------------------------------------
    # 恒温水浴锅 (Constant Temperature Water Bath)
    # -------------------------------------------------------------------------
    "water_bath": {
        "name": "恒温水浴锅",
        "name_en": "Constant Temperature Water Bath",
        "keywords": [
            "恒温", "水浴", "KXY", "数显", "°C", "℃",
            "SWITCH", "温度", "定时", "设定", "加热",
        ],
        "fields": {
            "temperature": {
                "chinese": "温度",
                "type": "number",
                "required": True,
                "unit": "℃",
            },
            "set_temperature": {
                "chinese": "设定温度",
                "type": "number",
                "unit": "℃",
            },
            "timer": {
                "chinese": "定时",
                "type": "string",
            },
        },
        "has_tabs": False,
    },
    
    # -------------------------------------------------------------------------
    # 表面张力仪 (Surface Tensiometer)
    # -------------------------------------------------------------------------
    "surface_tensiometer": {
        "name": "表面张力仪",
        "name_en": "Surface Tensiometer",
        "keywords": [
            "Surface tension", "表面张力", "界面张力", "mN/m",
            "温度", "密度", "g/cm", "TUI",
            "上升速度", "下降速度", "mm/min",
            "铂金板", "铂金环", "KRUSS",
        ],
        "fields": {
            "surface_tension": {
                "chinese": "表面/界面张力",
                "type": "number",
                "required": True,
                "unit": "mN/m",
            },
            "temperature": {
                "chinese": "温度",
                "type": "number",
                "unit": "℃",
            },
            "density": {
                "chinese": "密度",
                "type": "number",
                "unit": "g/cm³",
            },
            "rise_speed": {
                "chinese": "上升速度",
                "type": "number",
                "unit": "mm/min",
            },
            "fall_speed": {
                "chinese": "下降速度",
                "type": "number",
                "unit": "mm/min",
            },
            "timestamp": {
                "chinese": "时间戳",
                "type": "datetime",
            },
        },
        "has_tabs": False,
    },
    
    # -------------------------------------------------------------------------
    # 温湿度计 (Thermo-Hygrometer)
    # -------------------------------------------------------------------------
    "thermo_hygrometer": {
        "name": "温湿度计",
        "name_en": "Thermo-Hygrometer",
        "keywords": [
            "%RH", "湿度", "温度", "°C", "℃", "HI", "LO",
            "MAX", "MIN", "TEMP", "HUMIDITY", "露点",
        ],
        "fields": {
            "temperature": {
                "chinese": "温度",
                "type": "number",
                "required": True,
                "unit": "℃",
            },
            "humidity": {
                "chinese": "湿度",
                "type": "number",
                "required": True,
                "unit": "%RH",
            },
            "max_temp": {
                "chinese": "最高温度",
                "type": "number",
            },
            "min_temp": {
                "chinese": "最低温度",
                "type": "number",
            },
            "max_humidity": {
                "chinese": "最高湿度",
                "type": "number",
            },
            "min_humidity": {
                "chinese": "最低湿度",
                "type": "number",
            },
        },
        "has_tabs": False,
    },
    
    # -------------------------------------------------------------------------
    # pH计 (pH Meter)
    # -------------------------------------------------------------------------
    "ph_meter": {
        "name": "pH计",
        "name_en": "pH Meter",
        "keywords": [
            "pH", "PH", "mV", "CAL", "HOLD", "ATC",
            "酸度", "碱度", "缓冲液",
        ],
        "fields": {
            "ph_value": {
                "chinese": "pH值",
                "type": "number",
                "required": True,
            },
            "temperature": {
                "chinese": "温度",
                "type": "number",
                "unit": "℃",
            },
            "mv_value": {
                "chinese": "mV值",
                "type": "number",
                "unit": "mV",
            },
        },
        "has_tabs": False,
    },
    
    # -------------------------------------------------------------------------
    # 电导率仪 (Conductivity Meter)
    # -------------------------------------------------------------------------
    "conductivity_meter": {
        "name": "电导率仪",
        "name_en": "Conductivity Meter",
        "keywords": [
            "电导率", "μS/cm", "mS/cm", "TDS", "盐度",
            "conductivity", "COND",
        ],
        "fields": {
            "conductivity": {
                "chinese": "电导率",
                "type": "number",
                "required": True,
                "unit": "μS/cm",
            },
            "tds": {
                "chinese": "TDS",
                "type": "number",
                "unit": "ppm",
            },
            "salinity": {
                "chinese": "盐度",
                "type": "number",
                "unit": "%",
            },
            "temperature": {
                "chinese": "温度",
                "type": "number",
                "unit": "℃",
            },
        },
        "has_tabs": False,
    },
    
    # -------------------------------------------------------------------------
    # 溶解氧仪 (Dissolved Oxygen Meter)
    # -------------------------------------------------------------------------
    "do_meter": {
        "name": "溶解氧仪",
        "name_en": "Dissolved Oxygen Meter",
        "keywords": [
            "溶解氧", "DO", "mg/L", "饱和度", "%sat",
            "dissolved oxygen",
        ],
        "fields": {
            "do_value": {
                "chinese": "溶解氧",
                "type": "number",
                "required": True,
                "unit": "mg/L",
            },
            "saturation": {
                "chinese": "饱和度",
                "type": "number",
                "unit": "%",
            },
            "temperature": {
                "chinese": "温度",
                "type": "number",
                "unit": "℃",
            },
        },
        "has_tabs": False,
    },
}

# =============================================================================
# LLM 配置
# =============================================================================
LLM_CONFIG = {
    "lmstudio": {
        "default_url": "http://127.0.0.1:1234",
        "timeout": 60,
        "temperature": 0.1,
    },
    "ollama": {
        "default_url": "http://localhost:11434",
        "default_model": "qwen2:7b",
    },
}
