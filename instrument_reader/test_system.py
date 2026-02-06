#!/usr/bin/env python3
"""
测试脚本 - 验证系统各模块是否正常工作
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config():
    """测试配置加载"""
    print("\n" + "="*50)
    print("测试 1: 配置文件")
    print("="*50)
    
    try:
        from config import INSTRUMENT_CONFIG, OCR_RETRY_CONFIG
        
        print(f"✅ 配置加载成功")
        print(f"   支持的仪器类型: {len(INSTRUMENT_CONFIG)} 种")
        for inst_type, config in INSTRUMENT_CONFIG.items():
            print(f"   • {config['name']} ({inst_type})")
        print(f"   重试次数: {OCR_RETRY_CONFIG['max_retries']}")
        print(f"   置信度阈值: {OCR_RETRY_CONFIG['confidence_threshold']}")
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_ocr_module():
    """测试OCR模块"""
    print("\n" + "="*50)
    print("测试 2: OCR模块")
    print("="*50)
    
    try:
        from ocr_module import (
            create_ocr_engine, HAS_CV2, HAS_PIL,
            format_ocr_results_for_display, get_ocr_statistics,
            OCRResult
        )
        
        print(f"✅ OCR模块导入成功")
        print(f"   OpenCV可用: {HAS_CV2}")
        print(f"   Pillow可用: {HAS_PIL}")
        
        # 测试格式化函数
        test_results = [
            OCRResult("测试文本1", 0.95, [[0,0], [100,0], [100,30], [0,30]]),
            OCRResult("测试文本2", 0.80, [[0,30], [100,30], [100,60], [0,60]]),
        ]
        formatted = format_ocr_results_for_display(test_results)
        print(f"   格式化输出测试: OK")
        
        stats = get_ocr_statistics(test_results)
        print(f"   统计函数测试: 平均={stats['avg']:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ OCR模块测试失败: {e}")
        return False


def test_llm_module():
    """测试LLM模块"""
    print("\n" + "="*50)
    print("测试 3: LLM模块")
    print("="*50)
    
    try:
        from llm_module import (
            create_llm, ValidationResult, INSTRUMENT_SCHEMAS
        )
        
        print(f"✅ LLM模块导入成功")
        print(f"   加载的仪器模式: {len(INSTRUMENT_SCHEMAS)} 种")
        
        # 测试规则解析器
        parser = create_llm("rule")
        print(f"   规则解析器创建: OK")
        
        # 测试仪器识别
        test_ocr = [
            {"text": "检测结果", "confidence": 0.9},
            {"text": "吸光度", "confidence": 0.85},
            {"text": "0.000", "confidence": 0.9},
            {"text": "透光度", "confidence": 0.85},
            {"text": "100.00%", "confidence": 0.9},
        ]
        
        inst_type, conf = parser.identify_instrument_type(test_ocr)
        print(f"   仪器识别测试: {inst_type} (置信度: {conf:.2f})")
        
        # 测试解析
        result = parser.parse_instrument_reading(test_ocr)
        print(f"   解析测试: {result.status.value}")
        
        return True
    except Exception as e:
        print(f"❌ LLM模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_instrument_recognition():
    """测试仪器类型识别"""
    print("\n" + "="*50)
    print("测试 4: 仪器类型识别")
    print("="*50)
    
    try:
        from llm_module import create_llm
        
        parser = create_llm("rule")
        
        # 测试用例
        test_cases = [
            {
                "name": "水质检测仪",
                "expected": "water_quality_tester",
                "ocr": [
                    {"text": "检测结果", "confidence": 0.9},
                    {"text": "检测项目", "confidence": 0.85},
                    {"text": "吸光度", "confidence": 0.9},
                    {"text": "透光度", "confidence": 0.85},
                ]
            },
            {
                "name": "粘度计",
                "expected": "viscometer",
                "ocr": [
                    {"text": "速度:", "confidence": 0.9},
                    {"text": "0", "confidence": 0.95},
                    {"text": "粘度:", "confidence": 0.9},
                    {"text": "VISCOMETER", "confidence": 0.8},
                ]
            },
            {
                "name": "电子秤",
                "expected": "electronic_scale",
                "ocr": [
                    {"text": "60.39", "confidence": 0.95},
                    {"text": "g", "confidence": 0.9},
                    {"text": "HUAZHI", "confidence": 0.6},
                ]
            },
            {
                "name": "混调器",
                "expected": "mixer_stirrer",
                "ocr": [
                    {"text": "高速", "confidence": 0.9},
                    {"text": "转速(转)", "confidence": 0.85},
                    {"text": "3000", "confidence": 0.9},
                    {"text": "当前转速(转)", "confidence": 0.85},
                ]
            },
            {
                "name": "表面张力仪",
                "expected": "surface_tensiometer",
                "ocr": [
                    {"text": "Surface tension", "confidence": 0.8},
                    {"text": "mN/m", "confidence": 0.85},
                    {"text": "0.000", "confidence": 0.9},
                ]
            },
        ]
        
        passed = 0
        for case in test_cases:
            inst_type, conf = parser.identify_instrument_type(case["ocr"])
            status = "✅" if inst_type == case["expected"] else "❌"
            print(f"   {status} {case['name']}: {inst_type} (期望: {case['expected']})")
            if inst_type == case["expected"]:
                passed += 1
        
        print(f"\n   通过率: {passed}/{len(test_cases)}")
        return passed == len(test_cases)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_low_confidence():
    """测试低置信度处理"""
    print("\n" + "="*50)
    print("测试 5: 低置信度处理")
    print("="*50)
    
    try:
        from llm_module import create_llm
        
        parser = create_llm("rule")
        
        # 低置信度数据（平均0.25）
        low_conf_ocr = [
            {"text": "60.39", "confidence": 0.30},
            {"text": "g", "confidence": 0.20},
            {"text": "HUAZHI", "confidence": 0.25},
        ]
        
        avg_conf = sum(r["confidence"] for r in low_conf_ocr) / len(low_conf_ocr)
        print(f"   测试数据: 平均置信度 = {avg_conf:.2f}")
        
        is_valid, reason = parser.validate_ocr_results(low_conf_ocr)
        print(f"   验证结果: {'通过' if is_valid else '不通过'}")
        
        result = parser.parse_instrument_reading(low_conf_ocr)
        print(f"   解析状态: {result.status.value}")
        
        # 极低置信度（平均0.05）
        very_low_ocr = [
            {"text": "60.39", "confidence": 0.05},
            {"text": "g", "confidence": 0.05},
        ]
        
        avg_conf2 = sum(r["confidence"] for r in very_low_ocr) / len(very_low_ocr)
        print(f"\n   极低置信度测试: 平均 = {avg_conf2:.2f}")
        
        is_valid2, reason2 = parser.validate_ocr_results(very_low_ocr)
        print(f"   验证结果: {'通过' if is_valid2 else '不通过'} ({reason2})")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("   仪器读数识别系统 - 模块测试")
    print("="*60)
    
    results = {
        "配置文件": test_config(),
        "OCR模块": test_ocr_module(),
        "LLM模块": test_llm_module(),
        "仪器识别": test_instrument_recognition(),
        "低置信度": test_low_confidence(),
    }
    
    print("\n" + "="*60)
    print("   测试结果汇总")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n   总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统可以正常使用。")
        print("\n使用方法:")
        print("   python main.py -i photo.jpg")
        print("   python main.py -d ./images -o results.json")
    else:
        print("\n⚠️  部分测试失败，请检查相关模块。")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
