"""
仪器读数识别系统 - OCR模块
支持PaddleOCR，包含图像预处理和结果格式化
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# 可选依赖检查
# =============================================================================
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV未安装，部分预处理功能不可用: pip install opencv-python")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow未安装: pip install Pillow")


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    confidence: float
    box: List[List[int]]
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": self.box,
        }


# =============================================================================
# 图像预处理器
# =============================================================================
class ImagePreprocessor:
    """图像预处理工具类"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """加载图像文件"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        if HAS_CV2:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法加载图像: {image_path}")
            return img
        elif HAS_PIL:
            img = Image.open(image_path)
            return np.array(img.convert("RGB"))[:, :, ::-1]
        else:
            raise ImportError("需要安装OpenCV或Pillow: pip install opencv-python Pillow")
    
    @staticmethod
    def enhance_contrast(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """增强对比度（使用CLAHE算法）"""
        if not HAS_CV2:
            return img
        
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=factor * 2, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"对比度增强失败: {e}")
            return img
    
    @staticmethod
    def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
        """降噪处理"""
        if not HAS_CV2:
            return img
        
        try:
            return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
        except Exception as e:
            logger.warning(f"降噪处理失败: {e}")
            return img
    
    @staticmethod
    def binarize(img: np.ndarray, threshold: int = 127) -> np.ndarray:
        """二值化处理"""
        if not HAS_CV2:
            return img
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.warning(f"二值化处理失败: {e}")
            return img
    
    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        """锐化处理"""
        if not HAS_CV2:
            return img
        
        try:
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            return cv2.filter2D(img, -1, kernel)
        except Exception as e:
            logger.warning(f"锐化处理失败: {e}")
            return img
    
    @staticmethod
    def rotate(img: np.ndarray, angle: int) -> np.ndarray:
        """旋转图像（支持90°倍数）"""
        if angle == 0 or not HAS_CV2:
            return img
        
        try:
            if angle == 90:
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                return cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return img
        except Exception as e:
            logger.warning(f"旋转处理失败: {e}")
            return img
    
    @classmethod
    def preprocess(
        cls,
        img: np.ndarray,
        enhance_contrast: bool = False,
        contrast_factor: float = 1.5,
        denoise: bool = False,
        binarize: bool = False,
        binarize_threshold: int = 127,
        sharpen: bool = False,
        rotation: int = 0,
        **kwargs
    ) -> np.ndarray:
        """应用预处理流水线"""
        result = img.copy()
        
        if rotation != 0:
            result = cls.rotate(result, rotation)
        
        if enhance_contrast:
            result = cls.enhance_contrast(result, contrast_factor)
        
        if denoise:
            result = cls.denoise(result)
        
        if sharpen:
            result = cls.sharpen(result)
        
        if binarize:
            result = cls.binarize(result, binarize_threshold)
        
        return result


# =============================================================================
# OCR引擎基类
# =============================================================================
class OCREngine(ABC):
    """OCR引擎抽象基类"""
    
    @abstractmethod
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """识别图像中的文字"""
        pass
    
    def recognize_file(
        self,
        image_path: str,
        preprocess_config: Optional[Dict] = None,
    ) -> List[OCRResult]:
        """识别图像文件"""
        img = ImagePreprocessor.load_image(image_path)
        
        if preprocess_config:
            img = ImagePreprocessor.preprocess(img, **preprocess_config)
        
        return self.recognize(img)


# =============================================================================
# PaddleOCR引擎
# =============================================================================
class PaddleOCREngine(OCREngine):
    """PaddleOCR引擎实现（适配新版API）"""
    
    def __init__(
        self,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        use_gpu: bool = False,
    ):
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.use_gpu = use_gpu
        self._ocr = None
    
    def _lazy_init(self):
        """延迟初始化OCR引擎"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                import logging as _logging
                
                # 抑制PaddleOCR的日志输出
                _logging.getLogger("ppocr").setLevel(_logging.WARNING)
                
                self._ocr = PaddleOCR(
                    use_doc_orientation_classify=self.use_doc_orientation_classify,
                    use_doc_unwarping=self.use_doc_unwarping,
                    use_textline_orientation=self.use_textline_orientation,
                )
                
                logger.info("PaddleOCR引擎初始化成功")
            except ImportError:
                raise ImportError(
                    "PaddleOCR未安装，请执行: pip install paddlepaddle paddleocr"
                )
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """识别图像"""
        self._lazy_init()
        
        results = []
        try:
            # 新版API使用predict方法
            ocr_output = self._ocr.predict(input=image)
            
            if ocr_output:
                for res in ocr_output:
                    # 从结果对象中提取识别数据
                    if hasattr(res, 'rec_texts') and hasattr(res, 'rec_scores') and hasattr(res, 'dt_polys'):
                        texts = res.rec_texts
                        scores = res.rec_scores
                        boxes = res.dt_polys
                        
                        for text, score, box in zip(texts, scores, boxes):
                            if text and text.strip():
                                # 转换box格式
                                box_list = [[int(p[0]), int(p[1])] for p in box]
                                results.append(OCRResult(
                                    text=text.strip(),
                                    confidence=float(score),
                                    box=box_list,
                                ))
                    # 兼容字典格式的结果
                    elif isinstance(res, dict):
                        if 'rec_texts' in res:
                            texts = res.get('rec_texts', [])
                            scores = res.get('rec_scores', [])
                            boxes = res.get('dt_polys', [])
                            
                            for i, text in enumerate(texts):
                                if text and text.strip():
                                    score = scores[i] if i < len(scores) else 0.0
                                    box = boxes[i] if i < len(boxes) else [[0,0],[100,0],[100,30],[0,30]]
                                    box_list = [[int(p[0]), int(p[1])] for p in box]
                                    results.append(OCRResult(
                                        text=text.strip(),
                                        confidence=float(score),
                                        box=box_list,
                                    ))
        except Exception as e:
            logger.error(f"OCR识别出错: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def recognize_file(
        self,
        image_path: str,
        preprocess_config: Optional[Dict] = None,
    ) -> List[OCRResult]:
        """识别图像文件"""
        self._lazy_init()
        
        # 如果需要预处理，则加载图像进行处理
        if preprocess_config and any(preprocess_config.values()):
            img = ImagePreprocessor.load_image(image_path)
            img = ImagePreprocessor.preprocess(img, **preprocess_config)
            return self.recognize(img)
        
        # 否则直接使用文件路径（新版API支持）
        results = []
        try:
            ocr_output = self._ocr.predict(input=image_path)
            
            if ocr_output:
                for res in ocr_output:
                    if hasattr(res, 'rec_texts') and hasattr(res, 'rec_scores') and hasattr(res, 'dt_polys'):
                        texts = res.rec_texts
                        scores = res.rec_scores
                        boxes = res.dt_polys
                        
                        for text, score, box in zip(texts, scores, boxes):
                            if text and text.strip():
                                box_list = [[int(p[0]), int(p[1])] for p in box]
                                results.append(OCRResult(
                                    text=text.strip(),
                                    confidence=float(score),
                                    box=box_list,
                                ))
                    elif isinstance(res, dict):
                        if 'rec_texts' in res:
                            texts = res.get('rec_texts', [])
                            scores = res.get('rec_scores', [])
                            boxes = res.get('dt_polys', [])
                            
                            for i, text in enumerate(texts):
                                if text and text.strip():
                                    score = scores[i] if i < len(scores) else 0.0
                                    box = boxes[i] if i < len(boxes) else [[0,0],[100,0],[100,30],[0,30]]
                                    box_list = [[int(p[0]), int(p[1])] for p in box]
                                    results.append(OCRResult(
                                        text=text.strip(),
                                        confidence=float(score),
                                        box=box_list,
                                    ))
        except Exception as e:
            logger.error(f"OCR识别出错: {e}")
            import traceback
            traceback.print_exc()
        
        return results


# =============================================================================
# EasyOCR引擎（备选）
# =============================================================================
class EasyOCREngine(OCREngine):
    """EasyOCR引擎实现（备选方案）"""
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        self.languages = languages or ["ch_sim", "en"]
        self.use_gpu = use_gpu
        self._reader = None
    
    def _lazy_init(self):
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
                logger.info("EasyOCR引擎初始化成功")
            except ImportError:
                raise ImportError("EasyOCR未安装，请执行: pip install easyocr")
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        self._lazy_init()
        
        results = []
        try:
            if HAS_CV2 and len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            ocr_output = self._reader.readtext(image_rgb)
            
            for item in ocr_output:
                box = item[0]
                text = item[1]
                confidence = item[2]
                
                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    box=[[int(p[0]), int(p[1])] for p in box],
                ))
        except Exception as e:
            logger.error(f"EasyOCR识别出错: {e}")
        
        return results


# =============================================================================
# 模拟OCR引擎（用于测试）
# =============================================================================
class MockOCREngine(OCREngine):
    """模拟OCR引擎，用于开发测试"""
    
    def __init__(self, mock_results: List[Dict] = None):
        self.mock_results = mock_results or []
    
    def set_mock_results(self, results: List[Dict]):
        """设置模拟结果"""
        self.mock_results = results
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        return [
            OCRResult(
                text=r.get("text", ""),
                confidence=r.get("confidence", 0.9),
                box=r.get("box", [[0, 0], [100, 0], [100, 30], [0, 30]]),
            )
            for r in self.mock_results
        ]


# =============================================================================
# 工厂函数
# =============================================================================
def create_ocr_engine(engine_type: str = "paddle", **kwargs) -> OCREngine:
    """
    创建OCR引擎实例
    
    Args:
        engine_type: 引擎类型 ("paddle", "easyocr", "mock")
        **kwargs: 引擎特定参数
            - paddle: use_doc_orientation_classify, use_doc_unwarping, use_textline_orientation
            - easyocr: languages, use_gpu
    
    Returns:
        OCREngine实例
    """
    engines = {
        "paddle": PaddleOCREngine,
        "easyocr": EasyOCREngine,
        "mock": MockOCREngine,
    }
    
    if engine_type not in engines:
        raise ValueError(f"不支持的OCR引擎: {engine_type}，可选: {list(engines.keys())}")
    
    return engines[engine_type](**kwargs)


# =============================================================================
# 辅助函数
# =============================================================================
def format_ocr_results_for_display(results: List[OCRResult], max_width: int = 50) -> str:
    """
    格式化OCR结果用于终端显示
    
    Args:
        results: OCR结果列表
        max_width: 文本最大显示宽度
    
    Returns:
        格式化的字符串
    """
    if not results:
        return "  (无识别结果)"
    
    lines = []
    for i, r in enumerate(results, 1):
        text = r.text[:max_width] + "..." if len(r.text) > max_width else r.text
        # 置信度可视化条
        bar_filled = int(r.confidence * 10)
        conf_bar = "█" * bar_filled + "░" * (10 - bar_filled)
        lines.append(f"  [{i:2d}] {conf_bar} {r.confidence:.2f} │ {text}")
    
    return "\n".join(lines)


def get_ocr_statistics(results: List[OCRResult]) -> Dict[str, float]:
    """
    计算OCR结果统计信息
    
    Args:
        results: OCR结果列表
    
    Returns:
        统计信息字典
    """
    if not results:
        return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
    
    confidences = [r.confidence for r in results]
    return {
        "count": len(results),
        "avg": sum(confidences) / len(confidences),
        "min": min(confidences),
        "max": max(confidences),
    }
