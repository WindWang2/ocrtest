#!/usr/bin/env python3
"""
仪器读数识别系统 - 主程序
整合OCR识别、LLM解析、结果输出
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 确保模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_module import (
    create_ocr_engine, OCREngine,
    format_ocr_results_for_display, get_ocr_statistics
)
from llm_module import create_llm, LLMBase, ValidationResult, ParseResult
from config import OCR_RETRY_CONFIG, INSTRUMENT_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 仪器读数识别器
# =============================================================================
class InstrumentReader:
    """仪器读数识别器主类"""
    
    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        llm: Optional[LLMBase] = None,
        verbose: bool = True,
    ):
        """
        初始化识别器
        
        Args:
            ocr_engine: OCR引擎
            llm: LLM解析器
            verbose: 是否输出详细信息
        """
        self.ocr_engine = ocr_engine
        self.llm = llm
        self.verbose = verbose
        self.retry_config = OCR_RETRY_CONFIG
    
    def process_image(
        self,
        image_path: str,
        instrument_type: Optional[str] = None,
    ) -> Dict:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            instrument_type: 指定仪器类型（可选）
        
        Returns:
            处理结果字典
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📷 处理图片: {os.path.basename(image_path)}")
            print(f"{'='*60}")
        
        if not os.path.exists(image_path):
            return self._error_result(image_path, f"文件不存在: {image_path}")
        
        max_attempts = self.retry_config.get("max_retries", 3)
        retry_presets = self.retry_config.get("retry_presets", [{}])
        
        last_ocr_results = []
        last_error = ""
        
        for attempt in range(1, max_attempts + 1):
            if self.verbose:
                print(f"\n🔄 尝试 {attempt}/{max_attempts}")
            
            # 选择预处理配置
            preset_idx = min(attempt - 1, len(retry_presets) - 1)
            preprocess_config = retry_presets[preset_idx] if retry_presets else {}
            preset_name = preprocess_config.get("name", "default")
            
            if self.verbose:
                print(f"   预处理策略: {preset_name}")
            
            # OCR识别
            try:
                ocr_results = self.ocr_engine.recognize_file(
                    image_path,
                    preprocess_config=preprocess_config,
                )
            except Exception as e:
                logger.error(f"OCR识别失败: {e}")
                last_error = str(e)
                continue
            
            # 转换格式
            ocr_dicts = [r.to_dict() for r in ocr_results]
            last_ocr_results = ocr_dicts
            
            # 输出OCR结果
            if self.verbose:
                print(f"\n   📝 OCR识别结果 ({len(ocr_results)}条):")
                print(format_ocr_results_for_display(ocr_results))
                
                stats = get_ocr_statistics(ocr_results)
                if stats["count"] > 0:
                    print(f"\n   📊 置信度: 平均={stats['avg']:.2f}, "
                          f"最低={stats['min']:.2f}, 最高={stats['max']:.2f}")
            
            # LLM解析
            if self.verbose:
                print(f"\n   🤖 LLM解析中...")
            
            try:
                parse_result = self.llm.parse_instrument_reading(
                    ocr_dicts,
                    instrument_type=instrument_type,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
            except Exception as e:
                logger.error(f"LLM解析失败: {e}")
                last_error = str(e)
                continue
            
            # 处理结果
            if parse_result.status == ValidationResult.SUCCESS:
                if self.verbose:
                    inst_type = parse_result.data.get("instrument_type", "unknown")
                    inst_name = INSTRUMENT_CONFIG.get(inst_type, {}).get("name", inst_type)
                    print(f"\n   ✅ 识别成功!")
                    print(f"   仪器类型: {inst_name} ({inst_type})")
                    print(f"   置信度: {parse_result.confidence:.2%}")
                
                return {
                    "success": True,
                    "image_path": image_path,
                    "instrument_type": parse_result.data.get("instrument_type", "unknown"),
                    "readings": parse_result.data.get("readings", {}),
                    "confidence": parse_result.confidence,
                    "raw_ocr": parse_result.raw_ocr_texts,
                    "attempts": attempt,
                }
            
            elif parse_result.status == ValidationResult.NEED_RETRY:
                if self.verbose:
                    print(f"\n   ⚠️  需要重试: {parse_result.message}")
                last_error = parse_result.message
                continue
            
            else:  # FAILED
                if self.verbose:
                    print(f"\n   ❌ 解析失败: {parse_result.message}")
                last_error = parse_result.message
                break
        
        # 所有尝试失败
        return self._error_result(
            image_path, last_error,
            raw_ocr=[r.get("text", "") for r in last_ocr_results],
            attempts=max_attempts,
        )
    
    def process_directory(
        self,
        dir_path: str,
        recursive: bool = False,
        extensions: List[str] = None,
    ) -> List[Dict]:
        """
        处理目录中的所有图片
        
        Args:
            dir_path: 目录路径
            recursive: 是否递归处理子目录
            extensions: 支持的文件扩展名
        
        Returns:
            处理结果列表
        """
        extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]
        results = []
        
        path = Path(dir_path)
        if not path.exists():
            logger.error(f"目录不存在: {dir_path}")
            return results
        
        # 收集文件
        files = []
        for ext in extensions:
            if recursive:
                files.extend(path.rglob(f"*{ext}"))
            else:
                files.extend(path.glob(f"*{ext}"))
        
        files = sorted(set(files))
        
        if self.verbose:
            print(f"\n📁 找到 {len(files)} 个图片文件")
        
        for i, file_path in enumerate(files, 1):
            if self.verbose:
                print(f"\n[{i}/{len(files)}] ", end="")
            
            result = self.process_image(str(file_path))
            results.append(result)
        
        return results
    
    def _error_result(
        self,
        image_path: str,
        error: str,
        raw_ocr: List[str] = None,
        attempts: int = 0,
    ) -> Dict:
        """生成错误结果"""
        return {
            "success": False,
            "image_path": image_path,
            "error": error,
            "raw_ocr": raw_ocr or [],
            "attempts": attempts,
        }


# =============================================================================
# 结果输出
# =============================================================================
def print_summary(results: List[Dict]):
    """打印处理结果摘要"""
    print("\n" + "=" * 70)
    print("📊 处理结果汇总")
    print("=" * 70)
    
    total = len(results)
    success_count = sum(1 for r in results if r.get("success"))
    fail_count = total - success_count
    
    print(f"\n总计: {total} 张图片")
    print(f"✅ 成功: {success_count} ({success_count/total*100:.1f}%)" if total else "")
    print(f"❌ 失败: {fail_count} ({fail_count/total*100:.1f}%)" if total else "")
    
    # 按仪器类型分组
    by_type = {}
    for r in results:
        if r.get("success"):
            inst_type = r.get("instrument_type", "unknown")
            by_type.setdefault(inst_type, []).append(r)
    
    if by_type:
        print(f"\n📋 按仪器类型:")
        for inst_type, items in by_type.items():
            name = INSTRUMENT_CONFIG.get(inst_type, {}).get("name", inst_type)
            print(f"   • {name}: {len(items)} 张")
    
    # 详细结果
    print(f"\n{'─'*70}")
    print("详细结果:")
    print(f"{'─'*70}")
    
    for i, result in enumerate(results, 1):
        filename = os.path.basename(result.get("image_path", "unknown"))
        
        if result.get("success"):
            inst_type = result.get("instrument_type", "unknown")
            name = INSTRUMENT_CONFIG.get(inst_type, {}).get("name", inst_type)
            conf = result.get("confidence", 0)
            
            print(f"\n[{i}] ✅ {filename}")
            print(f"    仪器: {name}")
            print(f"    置信度: {conf:.2%}")
            print(f"    读数:")
            for key, value in result.get("readings", {}).items():
                print(f"      • {key}: {value}")
        else:
            print(f"\n[{i}] ❌ {filename}")
            print(f"    错误: {result.get('error', '未知错误')}")
            raw_ocr = result.get("raw_ocr", [])
            if raw_ocr:
                print(f"    原始OCR: {raw_ocr[:3]}{'...' if len(raw_ocr) > 3 else ''}")


# =============================================================================
# 主函数
# =============================================================================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="仪器读数识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单张图片
  python main.py -i photo.jpg
  
  # 处理目录
  python main.py -d ./images
  
  # 使用LM Studio大模型
  python main.py -i photo.jpg --llm lmstudio
  
  # 指定仪器类型
  python main.py -i photo.jpg -t water_quality_tester
  
  # 输出到JSON文件
  python main.py -d ./images -o results.json

支持的仪器类型:
  water_quality_tester  水质检测仪
  electronic_scale      电子秤
  viscometer           粘度计
  mixer_stirrer        混调器/搅拌器
  water_bath           恒温水浴锅
  surface_tensiometer  表面张力仪
  thermo_hygrometer    温湿度计
  ph_meter             pH计
  conductivity_meter   电导率仪
  do_meter             溶解氧仪
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--image", help="单张图片路径")
    input_group.add_argument("-d", "--dir", help="图片目录路径")
    
    # OCR选项
    parser.add_argument("--ocr", choices=["paddle", "easyocr"], default="paddle",
                       help="OCR引擎 (默认: paddle)")
    
    # LLM选项
    parser.add_argument("--llm", choices=["rule", "lmstudio", "ollama"], default="rule",
                       help="LLM类型 (默认: rule规则解析)")
    parser.add_argument("--lmstudio-url", default="http://127.0.0.1:1234",
                       help="LM Studio服务地址")
    parser.add_argument("--ollama-model", default="qwen2:7b",
                       help="Ollama模型名称")
    
    # 其他选项
    parser.add_argument("-t", "--type", help="指定仪器类型")
    parser.add_argument("-o", "--output", help="输出JSON文件路径")
    parser.add_argument("-r", "--recursive", action="store_true",
                       help="递归处理子目录")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="安静模式")
    
    args = parser.parse_args()
    
    # 创建OCR引擎
    print("🔧 初始化OCR引擎...")
    try:
        ocr_engine = create_ocr_engine(args.ocr)
    except Exception as e:
        logger.error(f"创建OCR引擎失败: {e}")
        print("\n💡 提示: 请安装PaddleOCR:")
        print("   pip install paddlepaddle paddleocr")
        return 1
    
    # 创建LLM
    print(f"🔧 初始化LLM解析器 ({args.llm})...")
    try:
        if args.llm == "lmstudio":
            llm = create_llm("lmstudio", base_url=args.lmstudio_url)
        elif args.llm == "ollama":
            llm = create_llm("ollama", model_name=args.ollama_model)
        else:
            llm = create_llm("rule")
    except Exception as e:
        logger.error(f"创建LLM失败: {e}")
        return 1
    
    # 创建识别器
    reader = InstrumentReader(
        ocr_engine=ocr_engine,
        llm=llm,
        verbose=not args.quiet,
    )
    
    # 处理图片
    results = []
    
    if args.image:
        result = reader.process_image(args.image, instrument_type=args.type)
        results = [result]
    elif args.dir:
        results = reader.process_directory(args.dir, recursive=args.recursive)
    
    # 打印摘要
    if results:
        print_summary(results)
    
    # 保存JSON
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "success": sum(1 for r in results if r.get("success")),
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
