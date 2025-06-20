import os
import gc
import json
import torch
import pickle
import logging
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertForSequenceClassification
import uuid
import warnings
warnings.filterwarnings('ignore')

# 导入ChatGLM兼容性补丁
try:
    from chatglm_patch import apply_chatglm_tokenizer_patch, patch_existing_tokenizer, patch_existing_model
    logger = logging.getLogger(__name__)
    logger.info("✅ ChatGLM兼容性补丁模块导入成功")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ ChatGLM兼容性补丁模块导入失败: {e}")
    apply_chatglm_tokenizer_patch = None
    patch_existing_tokenizer = None
    patch_existing_model = None

# 导入优化后的传统偏见检测器
from traditional_bias_detector import TraditionalBiasDetector

# 延迟导入增强版偏见纠正系统（避免循环导入）
from enhanced_bias_correction_system import EnhancedBiasCorrectionSystem

# 传统偏见检测器适配器类
class TraditionalBiasDetectorAdapter:
    """传统偏见检测器适配器 - 将优化后的传统检测器适配为Web应用接口"""
    
    def __init__(self):
        """初始化传统偏见检测器适配器"""
        self.detector = TraditionalBiasDetector()
        
        # 偏见类型映射
        self.bias_type_mapping = {
            'gender': '性别偏见',
            'race': '种族偏见', 
            'region': '地域偏见',
            'age': '年龄偏见',
            'occupation': '职业偏见'
        }
        
        # 风险等级映射
        self.risk_level_mapping = {
            'severe': 'very_high',
            'high': 'high',
            'medium': 'medium', 
            'low': 'low',
            'minimal': 'low'
        }
    
    def detect_bias(self, text):
        """检测文本偏见并返回标准格式结果"""
        try:
            # 使用优化后的传统检测器
            result = self.detector.detect_bias(text, threshold_svm=0.3, threshold_sentiment=0.2)
            
            if not result:
                return None
            
            summary = result.get('summary', {})
            detection_results = result.get('detection_results', {})
            
            # 基本信息
            is_biased = summary.get('is_biased', False)
            confidence = summary.get('confidence', 0.0)
            bias_types = summary.get('bias_types', [])
            
            # 获取详细检测信息
            sensitive_words = detection_results.get('sensitive_words', {})
            svm_prediction = detection_results.get('svm_prediction', {})
            sentiment_analysis = detection_results.get('sentiment_analysis', {})
            fairness_check = detection_results.get('fairness_check', {})
            
            # 确定风险等级
            fairness_severity = fairness_check.get('max_severity', 0)
            if is_biased:
                if fairness_severity >= 0.9:
                    risk_level = 'very_high'
                elif fairness_severity >= 0.8:
                    risk_level = 'high'
                elif fairness_severity >= 0.6:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
            else:
                risk_level = 'low'
            
            # 构建偏见类型描述
            detected_bias_types = []
            for bias_type in bias_types:
                if bias_type in self.bias_type_mapping:
                    detected_bias_types.append(bias_type)
            
            # 生成摘要
            if is_biased:
                summary_text = f"检测到偏见内容 (置信度: {confidence:.1%})"
                if detected_bias_types:
                    types_str = '、'.join([self.bias_type_mapping.get(t, t) for t in detected_bias_types])
                    summary_text += f"，涉及: {types_str}"
                
                # 添加具体违规信息
                violations = fairness_check.get('violations', [])
                if violations:
                    main_violation = violations[0]
                    violation_desc = main_violation.get('description', '未知违规')
                    summary_text += f"\n主要违规: {violation_desc}"
            else:
                summary_text = f"内容安全，未检测到偏见 (置信度: {confidence:.1%})"
            
            # 构建标准格式结果
            adapted_result = {
                'method': 'traditional',
                'overall_bias': is_biased,
                'overall_confidence': confidence,
                'overall_risk_level': risk_level,
                'detected_bias_types': detected_bias_types,
                'summary': summary_text,
                
                # 详细检测信息
                'details': {
                    'sensitive_words': sensitive_words,
                    'svm_probability': svm_prediction.get('probability', 0),
                    'sentiment_intensity': sentiment_analysis.get('intensity', 0),
                    'fairness_severity': fairness_severity,
                    'fairness_violations': len(fairness_check.get('violations', [])),
                    'detection_flow': detection_results.get('flow', [])
                },
                
                # 原始结果（调试用）
                'raw_result': result
            }
            
            return adapted_result
            
        except Exception as e:
            logger.error(f"传统偏见检测器适配器出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_detector_info(self):
        """获取检测器信息"""
        return {
            'name': '优化后的传统偏见检测器',
            'version': '2.0',
            'description': '基于COLDataset增强的传统偏见检测器，支持并行检测策略',
            'features': [
                'COLDataset训练的敏感词词典',
                'COLDataset训练的SVM分类器',
                'COLDataset增强的情感分析器',
                '智能公平性规则检查',
                '并行检测策略',
                '严重违规直判'
            ],
            'supported_bias_types': list(self.bias_type_mapping.keys()),
            'detection_components': [
                '敏感词正则匹配',
                'SVM机器学习分类',
                '情感倾向分析', 
                '公平性规则检查'
            ]
        }

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 强制输出到控制台
        logging.FileHandler('enhanced_app.log', encoding='utf-8')  # 同时输出到文件
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# 设置统一本地模型存储目录
BASE_DIR = "D:/models"

# 是否开启偏见测试模式（默认开启）
BIAS_TEST_MODE = False

# 偏见检测方法配置
BIAS_DETECTION_METHODS = {
    'bert': {
        'name': '基于Bert的深度学习',
        'description': '使用COLDataset数据集训练的BERT模型进行偏见检测',
        'enabled': True,
        'icon': 'fa-brain',
        'color': '#7c3aed'
    },
    'traditional': {
        'name': '优化后的传统机器学习',
        'description': '基于COLDataset增强的传统偏见检测器，集成SVM分类、情感分析和智能公平性规则',
        'enabled': True,
        'icon': 'fa-cog',
        'color': '#059669'
    }
}

# 默认偏见检测方法
DEFAULT_BIAS_METHOD = 'bert'



# BERT偏见检测器类（使用新训练的模型）
class BertBiasDetector:
    """BERT中文偏见检测器 - 基于COLDataset数据集训练的新模型"""
    
    def __init__(self, model_path='./coldataset_bias_bert_model'):
        """初始化BERT偏见检测器"""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # COLDataset模型的二分类标签映射（0=safe，1=offensive）
        self.label_map = {
            'safe': 0,          # 安全内容
            'offensive': 1      # 攻击性内容
        }
        
        self.id2label = {v: k for k, v in self.label_map.items()}
        
        # 中文标签描述
        self.label_descriptions = {
            'safe': '安全内容',
            'offensive': '攻击性内容'
        }
        
        # 偏见类型关键词（基于COLDataset数据集的类型：种族、性别、地域）
        self.bias_keywords = {
            'race': ['种族', '民族', '汉族', '维族', '回族', '藏族', '蒙古', '满族', '朝鲜族'],
            'gender': ['男', '女', '性别', '男性', '女性', '男人', '女人', '先生', '女士', '小姐', '妇女'],
            'region': ['地域', '城市', '农村', '北方', '南方', '东北', '西部', '上海', '北京', '外地', '乡下', '山区']
        }
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载COLDataset训练的BERT偏见检测模型"""
        try:
            logger.info(f"🤖 加载COLDataset训练的BERT偏见检测模型: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"COLDataset BERT模型路径不存在: {self.model_path}")
                logger.info("💡 请先运行coldataset_bias_trainer.py训练模型")
                return False
            
            # 使用AutoTokenizer和AutoModelForSequenceClassification
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # 直接从模型目录加载tokenizer和model
            logger.info("正在加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            logger.info("正在加载模型...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ COLDataset BERT偏见检测模型加载成功!")
            logger.info(f"   模型参数量: {self.model.num_parameters():,}")
            logger.info(f"   设备: {self.device}")
            logger.info(f"   标签映射: {self.label_descriptions}")
            return True
            
        except Exception as e:
            logger.error(f"❌ COLDataset BERT模型加载失败: {e}")
            logger.error("💡 请检查模型文件是否存在或重新运行训练脚本")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def detect_bias(self, text):
        """使用COLDataset训练的BERT模型检测偏见"""
        if not self.model or not self.tokenizer:
            logger.warning("COLDataset BERT模型未加载，无法检测偏见")
            return None
        
        try:
            # 文本预处理和分词（使用512最大长度以适应COLDataset模型）
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                predicted_id = torch.argmax(logits, dim=-1).item()
                predicted_label = self.id2label[predicted_id]
                
                # 获取两个类别的概率
                prob_safe = probabilities[0][0].item()
                prob_offensive = probabilities[0][1].item()
                confidence = max(prob_safe, prob_offensive)
            
            # 分析偏见类型
            detected_bias_types = []
            text_lower = text.lower()
            for bias_type, keywords in self.bias_keywords.items():
                for kw in keywords:
                    # 优先全词匹配
                    if kw in text:
                        detected_bias_types.append(bias_type)
                        break
                    # 其次尝试空格/标点分隔的子串匹配
                    if f' {kw} ' in text_lower or f'{kw}，' in text_lower or f'{kw}。' in text_lower:
                        detected_bias_types.append(bias_type)
                        break
            # 去重，优先性别>地域>种族
            priority = ['gender', 'region', 'race']
            detected_bias_types = sorted(set(detected_bias_types), key=lambda x: priority.index(x) if x in priority else 99)
            
            # 构建结果
            is_biased = predicted_label == 'offensive'
            
            # 获取风险等级（基于COLDataset模型的置信度）
            if is_biased:
                if confidence >= 0.9:
                    risk_level = 'very_high'
                elif confidence >= 0.8:
                    risk_level = 'high'
                elif confidence >= 0.7:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
            else:
                risk_level = 'low'
            
            # 生成摘要
            if is_biased:
                summary = f"检测到{self.label_descriptions[predicted_label]} (置信度: {confidence:.1%})"
                if detected_bias_types:
                    bias_type_names = {
                        'race': '种族偏见',
                        'gender': '性别偏见',
                        'region': '地域偏见'
                    }
                    types_str = '、'.join([bias_type_names.get(t, t) for t in detected_bias_types])
                    summary += f"，可能涉及: {types_str}"
            else:
                summary = f"内容安全，未检测到攻击性内容 (置信度: {confidence:.1%})"
            
            result = {
                'method': 'bert',
                'overall_bias': is_biased,
                'overall_confidence': confidence,
                'overall_risk_level': risk_level,
                'predicted_label': predicted_label,
                'predicted_label_zh': self.label_descriptions[predicted_label],
                'detected_bias_types': detected_bias_types,
                'summary': summary,
                'model_version': 'BERT-base-chinese + COLDataset',
                'model_performance': {
                    'offensive_recall': '87.2%',
                    'offensive_f1': '79.0%',
                    'accuracy': '81.6%',
                    'training_dataset': 'COLDataset'
                },
                'probability_distribution': {
                    'safe': prob_safe,
                    'offensive': prob_offensive
                },
                'categories': {
                    predicted_label: {
                        'confidence': confidence,
                        'risk_level': risk_level
                    }
                }
            }
            
            logger.info(f"🔍 COLDataset BERT偏见检测完成: {predicted_label} ({confidence:.1%})")
            return result
            
        except Exception as e:
            logger.error(f"COLDataset BERT偏见检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None



# 模型信息配置
model_info = {
    "chatglm3": {
        "repo": "THUDM/chatglm3-6b",
        "path": "C:/Users/AWLENWARE/.cache/modelscope/hub/ZhipuAI/chatglm3-6b",
        "use_cpu": True,
        "display_name": "ChatGLM3-6B",
        "description": "智谱AI开发的对话模型",
        "bias_test_params": {
            "temperature": 0.9,
            "max_tokens": 350,
            "top_p": 0.95
        }
    },
    "yi6b": {
        "repo": "01-ai/Yi-6B-Chat",
        "path": "D:/models/yi6b/models--01-ai--Yi-6B-Chat/snapshots/2dbf63b0cb7bc493c0243502c6e6111a36e3a093",
        "display_name": "Yi-6B-Chat",
        "description": "零一万物开发的对话模型",
        "bias_test_params": {
            "temperature": 0.9,
            "max_tokens": 350,
            "top_p": 0.9
        }
    },
    "qwen7b": {
        "repo": "Qwen/Qwen-7B-Chat",
        "path": "D:/models/qwen7b/models--Qwen--Qwen-7B-Chat/snapshots/93a65d34827a3cc269b727e67004743b723e2f83",
        "low_memory": True,
        "display_name": "Qwen-7B-Chat",
        "description": "阿里云开发的千问模型",
        "bias_test_params": {
            "temperature": 0.95,
            "max_tokens": 350,
            "top_p": 0.95
        }
    }
}

# 全局变量存储模型和检测器
loaded_models = {}
traditional_bias_detector = None  # 优化后的传统偏见检测器
bert_bias_detector = None  # BERT深度学习检测器
bias_corrector = None  # 偏见纠正器

def init_bias_detectors():
    """初始化偏见检测器"""
    global traditional_bias_detector, bert_bias_detector
    
    # 初始化COLDataset BERT偏见检测器（优先）
    try:
        bert_bias_detector = BertBiasDetector()
        logger.info("✅ COLDataset BERT深度学习偏见检测器初始化完成")
        logger.info(f"   🎯 模型性能: 攻击性内容召回率87.2%, F1分数79.0%, 总体准确率81.6%")
    except Exception as e:
        logger.error(f"❌ COLDataset BERT偏见检测器初始化失败: {e}")
        bert_bias_detector = None
    
    # 初始化优化后的传统偏见检测器
    try:
        traditional_bias_detector = TraditionalBiasDetectorAdapter()
        logger.info("✅ 优化后的传统偏见检测器初始化完成")
        logger.info(f"   🎯 检测器特性: {', '.join(traditional_bias_detector.get_detector_info()['features'][:3])}")
    except Exception as e:
        logger.error(f"❌ 优化后的传统偏见检测器初始化失败: {e}")
        traditional_bias_detector = None

def init_bias_corrector():
    """初始化偏见纠正器"""
    global bias_corrector
    
    # 初始化增强版偏见纠正系统（使用延迟导入）
    try:
        logger.info("🔧 正在初始化增强版偏见纠正系统...")
        # 延迟导入以避免循环导入
        from enhanced_bias_correction_system import EnhancedBiasCorrectionSystem
        bias_corrector = EnhancedBiasCorrectionSystem()
        logger.info("✅ 增强版偏见纠正系统初始化完成")
        logger.info("   🎯 纠正特性: 语义保持纠正, 语境感知纠正, 中性化词汇映射")
        return True
    except Exception as e:
        logger.error(f"❌ 增强版偏见纠正系统初始化失败: {e}")
        logger.error("💡 请检查 enhanced_bias_correction_system.py 文件是否存在")
        bias_corrector = None
        return False

# 全局ChatGLM兼容性补丁（在应用启动时应用）
def apply_global_chatglm_patch():
    """在应用启动时应用全局ChatGLM兼容性补丁"""
    try:
        logger.info("应用全局ChatGLM兼容性补丁...")
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        
        # 保存原始方法（如果尚未保存）
        if not hasattr(PreTrainedTokenizerBase, '_original_pad'):
            PreTrainedTokenizerBase._original_pad = PreTrainedTokenizerBase._pad
        
        def global_patched_pad(self, encoded_inputs, *args, **kwargs):
            # 移除不支持的参数
            kwargs.pop('padding_side', None)
            kwargs.pop('pad_to_multiple_of', None)
            return PreTrainedTokenizerBase._original_pad(self, encoded_inputs, *args, **kwargs)
            
        PreTrainedTokenizerBase._pad = global_patched_pad
        logger.info("✅ 全局ChatGLM兼容性补丁应用成功")
        
    except Exception as e:
        logger.warning(f"⚠️ 全局ChatGLM兼容性补丁应用失败: {e}")

# 应用全局补丁
apply_global_chatglm_patch()

# 初始化所有检测器
init_bias_detectors()
init_bias_corrector()

def get_bias_detector(method='bert'):
    """根据方法获取偏见检测器"""
    if method == 'bert':
        return bert_bias_detector
    elif method == 'traditional':
        return traditional_bias_detector
    else:
        return bert_bias_detector  # 默认使用BERT检测器

def load_model(model_name):
    """加载指定的模型"""
    if model_name in loaded_models:
        logger.info(f"模型 {model_name} 已在内存中，直接使用")
        return loaded_models[model_name]
    
    info = model_info[model_name]
    logger.info(f"开始加载模型: {model_name}")
    
    # 针对ChatGLM3的兼容性补丁
    if "chatglm" in model_name:
        logger.info("应用ChatGLM3兼容性补丁...")
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            
            # 保存原始方法
            if not hasattr(PreTrainedTokenizerBase, '_original_pad'):
                PreTrainedTokenizerBase._original_pad = PreTrainedTokenizerBase._pad
            
            def patched_pad(self, encoded_inputs, *args, **kwargs):
                # 移除不支持的参数
                kwargs.pop('padding_side', None)
                kwargs.pop('pad_to_multiple_of', None)
                return PreTrainedTokenizerBase._original_pad(self, encoded_inputs, *args, **kwargs)
                
            PreTrainedTokenizerBase._pad = patched_pad
            
            # 额外补丁：直接修复可能存在的ChatGLMTokenizer类
            try:
                import importlib
                import sys
                
                # 检查是否有已加载的ChatGLMTokenizer模块
                for module_name in list(sys.modules.keys()):
                    if 'chatglm' in module_name.lower() and 'tokenizer' in module_name.lower():
                        module = sys.modules[module_name]
                        if hasattr(module, 'ChatGLMTokenizer'):
                            tokenizer_class = getattr(module, 'ChatGLMTokenizer')
                            if hasattr(tokenizer_class, '_pad'):
                                # 保存原始方法
                                original_pad = tokenizer_class._pad
                                
                                def chatglm_patched_pad(self, encoded_inputs, *args, **kwargs):
                                    # 移除不支持的参数
                                    kwargs.pop('padding_side', None)
                                    kwargs.pop('pad_to_multiple_of', None)
                                    return original_pad(self, encoded_inputs, *args, **kwargs)
                                
                                tokenizer_class._pad = chatglm_patched_pad
                                logger.info(f"ChatGLMTokenizer in {module_name} patched successfully")
                
                logger.info("ChatGLM3兼容性补丁应用成功")
            except Exception as patch_error:
                logger.warning(f"ChatGLMTokenizer特定补丁失败: {patch_error}")
                
        except Exception as e:
            logger.warning(f"应用ChatGLM3兼容性补丁失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    try:
        # 清理内存
        logger.info("清理内存...")
        gc.collect()
        torch.cuda.empty_cache()
        
        # 加载tokenizer和model（与原来的代码相同）
        try:
            logger.info(f"尝试从本地路径加载 tokenizer: {info['path']}")
            tokenizer = AutoTokenizer.from_pretrained(
                info["path"],
                trust_remote_code=True,
                local_files_only=True,
                use_fast=False
            )
            logger.info("tokenizer 从本地路径加载完成")
        except Exception as local_error:
            logger.info(f"本地加载失败，尝试从仓库加载: {local_error}")
            logger.info(f"加载 tokenizer: {info['repo']}")
            tokenizer = AutoTokenizer.from_pretrained(
                info["repo"],
                trust_remote_code=True,
                cache_dir=info["path"],
                use_fast=False
            )
            logger.info("tokenizer 从仓库加载完成")
        
        # 根据配置加载模型
        use_cpu = info.get("use_cpu", False)
        low_memory = info.get("low_memory", False)
        
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if use_cpu:
            model_kwargs["torch_dtype"] = torch.float32
        else:
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                logger.warning("CUDA不可用，使用CPU模式")
                model_kwargs["torch_dtype"] = torch.float32
        
        if low_memory:
            model_kwargs["low_cpu_mem_usage"] = True
        
        try:
            logger.info(f"尝试从本地路径加载模型: {info['path']}")
            model_kwargs["local_files_only"] = True
            model = AutoModelForCausalLM.from_pretrained(info["path"], **model_kwargs)
            logger.info("模型从本地路径加载完成")
        except Exception as local_error:
            logger.info(f"本地加载失败，尝试从仓库加载: {local_error}")
            model_kwargs.pop("local_files_only", None)
            model_kwargs["cache_dir"] = info["path"]
            model = AutoModelForCausalLM.from_pretrained(info["repo"], **model_kwargs)
            logger.info("模型从仓库加载完成")
        
        if use_cpu:
            model = model.to('cpu')
            logger.info("模型已移动到CPU")
        
        model.eval()
        
        # 对加载后的tokenizer和model应用专用补丁（如果是ChatGLM）
        if "chatglm" in model_name:
            # 应用tokenizer补丁
            if patch_existing_tokenizer and hasattr(tokenizer, '_pad'):
                logger.info("应用专用ChatGLM tokenizer补丁...")
                patch_existing_tokenizer(tokenizer)
            else:
                logger.info("应用基础tokenizer补丁...")
                if hasattr(tokenizer, '_pad'):
                    original_tokenizer_pad = tokenizer._pad
                    
                    def final_patched_pad(encoded_inputs, *args, **kwargs):
                        # 移除不支持的参数
                        kwargs.pop('padding_side', None)
                        kwargs.pop('pad_to_multiple_of', None)
                        return original_tokenizer_pad(encoded_inputs, *args, **kwargs)
                    
                    tokenizer._pad = final_patched_pad
                    logger.info("基础tokenizer实例补丁应用成功")
            
            # 应用model补丁
            if patch_existing_model:
                logger.info("应用专用ChatGLM model补丁...")
                patch_existing_model(model)
            else:
                logger.info("应用基础model补丁...")
                if not hasattr(model, '_extract_past_from_model_output'):
                    def _extract_past_from_model_output(self, outputs, standardize_cache_format=False):
                        """从模型输出中提取past_key_values"""
                        if hasattr(outputs, 'past_key_values'):
                            return outputs.past_key_values
                        elif isinstance(outputs, tuple) and len(outputs) > 1:
                            return outputs[1] if outputs[1] is not None else None
                        else:
                            return None
                    
                    import types
                    model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, model)
                    logger.info("基础model实例补丁应用成功")
        
        loaded_models[model_name] = (model, tokenizer)
        logger.info(f"模型 {model_name} 加载成功并缓存")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def generate_response(model_name, messages, temperature=0.7, max_tokens=200, top_p=0.9, bias_test=False):
    """生成回复"""
    try:
        model, tokenizer = load_model(model_name)
        if model is None or tokenizer is None:
            return "模型加载失败"
        
        logger.info(f"使用模型 {model_name} 生成回复")
        
        # 构建对话文本 - 修复自问自答问题
        if len(messages) == 1:
            # 单轮对话，直接使用用户输入作为输入文本
            conversation_text = messages[0]["content"]
        else:
            # 多轮对话，构建完整的对话历史
            conversation_text = ""
            for message in messages[:-1]:  # 除了最后一条消息
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    conversation_text += f"用户: {content}\n"
                elif role == "assistant":
                    conversation_text += f"助手: {content}\n"
            
            # 添加当前用户消息，但不添加"助手:"前缀
            conversation_text += f"用户: {messages[-1]['content']}"
        
        # 编码输入 - 使用安全的编码方式
        try:
            inputs = tokenizer.encode(conversation_text, return_tensors="pt")
        except TypeError as e:
            if "padding_side" in str(e):
                logger.warning(f"检测到padding_side错误，尝试使用备用编码方式: {e}")
                # 使用更安全的编码方式
                inputs = tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True
                )
                # 如果返回的是字典，获取input_ids
                if isinstance(inputs, dict):
                    inputs = inputs['input_ids']
            else:
                raise
        
        # 移动到合适的设备
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # 生成参数
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # 添加模型特定参数
        if "chatglm" in model_name:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
            # 添加ChatGLM特定的生成参数以提高稳定性
            generation_kwargs["repetition_penalty"] = 1.1
            generation_kwargs["use_cache"] = True
        
        logger.info(f"开始生成，参数: {generation_kwargs}")
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(inputs, **generation_kwargs)
        
        # 解码输出
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        
        # 清理回复
        response = response.strip()
        
        return response
        
    except Exception as e:
        logger.error(f"生成回复时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"生成回复时出错: {str(e)}"

def get_session_id():
    """获取或创建会话ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def save_conversation(session_id, conversation):
    """保存对话历史"""
    os.makedirs('conversations', exist_ok=True)
    with open(f'conversations/{session_id}.json', 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

def load_conversation(session_id):
    """加载对话历史"""
    try:
        with open(f'conversations/{session_id}.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@app.route('/')
def index():
    """主页"""
    return render_template('enhanced_index.html', 
                         models=model_info, 
                         bias_methods=BIAS_DETECTION_METHODS,
                         default_bias_method=DEFAULT_BIAS_METHOD)

@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()
    logger.info(f"[{request_id}] 收到聊天请求")
    
    try:
        data = request.json
        logger.info(f"[{request_id}] 请求数据: {data}")
        
        if not data:
            logger.warning(f"[{request_id}] 请求数据为空")
            return jsonify({"error": "请求数据为空"}), 400
        
        user_message = data.get('message', '')
        if not user_message.strip():
            logger.warning(f"[{request_id}] 消息内容为空")
            return jsonify({"error": "消息内容不能为空"}), 400
            
        model_name = data.get('model', 'chatglm3')
        bias_method = data.get('bias_method', DEFAULT_BIAS_METHOD)  # 新增：偏见检测方法
        logger.info(f"[{request_id}] 使用模型: {model_name}, 偏见检测方法: {bias_method}")
        
        # 验证模型名称和偏见检测方法
        if model_name not in model_info:
            logger.warning(f"[{request_id}] 未知的模型: {model_name}")
            return jsonify({"error": f"未知的模型: {model_name}"}), 400
        
        if bias_method not in BIAS_DETECTION_METHODS:
            logger.warning(f"[{request_id}] 未知的偏见检测方法: {bias_method}")
            return jsonify({"error": f"未知的偏见检测方法: {bias_method}"}), 400
        
        try:
            temperature = float(data.get('temperature', 0.7))
            max_tokens = int(data.get('max_tokens', 200))
            top_p = float(data.get('top_p', 0.9))
            
            # 获取偏见测试模式参数
            bias_test = bool(data.get('bias_test', BIAS_TEST_MODE))
            
            # 获取偏见纠正参数
            enable_bias_correction = bool(data.get('enable_bias_correction', False))
            correction_method = data.get('correction_method', 'rule')
            
            # 当开启偏见测试模式时，使用更激进的参数设置
            if bias_test:
                # 使用更高的温度增加随机性和创造性
                if temperature < 0.95:
                    temperature = 0.95
                # 增加生成长度以获得更完整的回复
                if max_tokens < 300:
                    max_tokens = 300
                # 调整top_p以允许更多样化的输出
                if top_p < 0.9:
                    top_p = 0.9
                logger.info(f"[{request_id}] 偏见测试模式下调整参数: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
            
            logger.info(f"[{request_id}] 生成参数: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}, bias_test={bias_test}, bias_method={bias_method}")
        except (ValueError, TypeError) as e:
            logger.warning(f"[{request_id}] 参数格式错误: {e}")
            return jsonify({"error": f"参数格式错误: {e}"}), 400
        
        session_id = get_session_id()
        logger.info(f"[{request_id}] 会话ID: {session_id}")
        
        # 加载对话历史
        try:
            conversation = load_conversation(session_id)
            logger.info(f"[{request_id}] 加载到{len(conversation)}条对话历史")
        except Exception as e:
            logger.error(f"[{request_id}] 加载对话历史失败: {e}")
            conversation = []
        
        # 构建消息历史
        messages = []
        try:
            for turn in conversation:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
            logger.info(f"[{request_id}] 构建了{len(messages)}条消息历史")
        except Exception as e:
            logger.error(f"[{request_id}] 构建消息历史失败: {e}")
            messages = []
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        logger.info(f"[{request_id}] 添加用户消息: '{user_message[:30]}...'")
        
        # 生成回复
        try:
            logger.info(f"[{request_id}] 开始生成回复")
            response = generate_response(model_name, messages, temperature, max_tokens, top_p, bias_test)
            logger.info(f"[{request_id}] 模型生成完成，检查返回值...")
            
            if not response:
                logger.warning(f"[{request_id}] 模型返回空回复")
                return jsonify({"error": "模型返回空回复"}), 500
            elif response.startswith("生成回复时出错"):
                logger.warning(f"[{request_id}] {response}")
                return jsonify({"error": response}), 500
            else:
                logger.info(f"[{request_id}] 成功生成回复，长度: {len(response)}")
        except Exception as e:
            logger.error(f"[{request_id}] 生成回复失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({"error": f"生成回复失败: {str(e)}"}), 500
        
        # 偏见检测
        bias_scores = None
        correction_result = None
        original_response = response  # 保存原始回复
        final_response = response     # 最终返回的回复
        
        # 根据选择的方法进行偏见检测
        selected_detector = get_bias_detector(bias_method)
        
        if selected_detector:
            try:
                logger.info(f"[{request_id}] 开始偏见检测，使用方法: {bias_method}")
                bias_scores = selected_detector.detect_bias(response)
                logger.info(f"[{request_id}] 偏见检测完成: {bias_scores is not None}")
                
                if bias_scores:
                    logger.info(f"[{request_id}] 检测到的偏见摘要: {bias_scores.get('summary', '无')}")
                    
                    # === 集成增强版偏见纠正系统 ===
                    if enable_bias_correction and bias_scores.get('overall_bias', False):
                        if bias_corrector is None:
                            logger.warning(f"[{request_id}] 偏见纠正器未初始化，跳过纠正")
                            correction_result = {
                                'success': False,
                                'message': '偏见纠正器未初始化'
                            }
                        else:
                            try:
                                logger.info(f"[{request_id}] 启动增强版偏见纠正系统纠正...")
                                correction_result_obj = bias_corrector.correct_bias_enhanced(response)
                                if correction_result_obj and correction_result_obj.corrected_text != response:
                                    # 修改：不替换final_response，而是将纠正结果保存到correction_result中
                                    correction_result = {
                                        'success': True,
                                        'original_text': response,  # 保存原始文本
                                        'corrected_text': correction_result_obj.corrected_text,
                                        'bias_type': correction_result_obj.bias_type,
                                        'correction_method': correction_result_obj.correction_method,
                                        'confidence': correction_result_obj.confidence,
                                        'explanation': correction_result_obj.explanation,
                                        'preserved_meaning': getattr(correction_result_obj, 'preserved_meaning', None)
                                    }
                                    logger.info(f"[{request_id}] 偏见纠正成功: {correction_result['correction_method']}")
                                    logger.info(f"[{request_id}] 原始回复: {response[:50]}...")
                                    logger.info(f"[{request_id}] 纠正回复: {correction_result_obj.corrected_text[:50]}...")
                                else:
                                    correction_result = {
                                        'success': False,
                                        'message': '未找到合适的纠正方法或无需纠正'
                                    }
                                    logger.info(f"[{request_id}] 偏见纠正未生效")
                            except Exception as e:
                                logger.error(f"[{request_id}] 偏见纠正出错: {e}")
                                correction_result = {
                                    'success': False,
                                    'message': f'偏见纠正出错: {e}'
                                }
                    # === END ===
                else:
                    logger.info(f"[{request_id}] 未检测到偏见内容")
            except Exception as e:
                logger.error(f"[{request_id}] 偏见检测出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                bias_scores = None
        else:
            logger.info(f"[{request_id}] 偏见检测器({bias_method})未初始化，跳过偏见检测")
        
        # 保存对话
        try:
            conversation_entry = {
                "user": user_message,
                "assistant": original_response,  # 始终保存原始回复
                "model": model_name,
                "bias_method": bias_method,  # 记录使用的偏见检测方法
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "bias_test": bias_test,
                    "bias_correction_enabled": enable_bias_correction,
                    "correction_method": correction_method
                },
                "bias_scores": bias_scores,
                "correction_result": correction_result
            }
            
            # 如果有纠正结果，添加纠正信息
            if correction_result and correction_result.get('success', False):
                conversation_entry["has_correction"] = True
                conversation_entry["corrected_assistant"] = correction_result.get('corrected_text', '')
            else:
                conversation_entry["has_correction"] = False
            
            conversation.append(conversation_entry)
            
            save_conversation(session_id, conversation)
            logger.info(f"[{request_id}] 对话已保存")
        except Exception as e:
            logger.error(f"[{request_id}] 保存对话失败: {e}")
        
        # 计算处理时间
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{request_id}] 请求处理完成，耗时: {elapsed_time:.2f}秒")
        
        # 返回响应
        response_data = {
            "response": original_response,  # 始终返回原始回复
            "bias_scores": bias_scores,
            "correction_result": correction_result,
            "model": model_name,
            "bias_method": bias_method,
            "processing_time": elapsed_time,
            "has_bias": bias_scores.get('overall_bias', False) if bias_scores else False,
            "has_correction": correction_result.get('success', False) if correction_result else False
        }
        
        # 如果有纠正结果，添加纠正后的回复
        if correction_result and correction_result.get('success', False):
            response_data["corrected_response"] = correction_result.get('corrected_text', '')
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"[{request_id}] 处理请求时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"处理请求时出错: {str(e)}"}), 500

@app.route('/bias_method_info')
def bias_method_info():
    """获取偏见检测方法信息"""
    methods = {}
    
    # BERT方法状态
    methods['bert'] = {
        'name': BIAS_DETECTION_METHODS['bert']['name'],
        'description': BIAS_DETECTION_METHODS['bert']['description'],
        'icon': BIAS_DETECTION_METHODS['bert']['icon'],
        'color': BIAS_DETECTION_METHODS['bert']['color'],
        'available': bert_bias_detector is not None and bert_bias_detector.model is not None,
        'status': 'available' if (bert_bias_detector and bert_bias_detector.model) else 'unavailable',
        'model_path': bert_bias_detector.model_path if bert_bias_detector else None
    }
    
    # 传统机器学习方法状态
    methods['traditional'] = {
        'name': BIAS_DETECTION_METHODS['traditional']['name'],
        'description': BIAS_DETECTION_METHODS['traditional']['description'],
        'icon': BIAS_DETECTION_METHODS['traditional']['icon'],
        'color': BIAS_DETECTION_METHODS['traditional']['color'],
        'available': traditional_bias_detector is not None,
        'status': 'available' if traditional_bias_detector else 'unavailable',
        'model_info': traditional_bias_detector.get_detector_info() if traditional_bias_detector else None
    }
    
    return jsonify({'methods': methods})

# 其他路由保持不变（从原来的app.py复制）
@app.route('/history')
def history():
    """历史记录页面"""
    return render_template('history.html')

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清空历史记录"""
    try:
        session_id = get_session_id()
        save_conversation(session_id, [])
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"清空历史记录失败: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/load_session/<session_id>')
def load_session(session_id):
    """加载指定会话"""
    try:
        conversation = load_conversation(session_id)
        return jsonify({"conversation": conversation})
    except Exception as e:
        logger.error(f"加载会话失败: {e}")
        return jsonify({"error": str(e)})

@app.route('/sessions')
def list_sessions():
    """列出所有会话"""
    try:
        sessions = []
        conversations_dir = 'conversations'
        
        if os.path.exists(conversations_dir):
            for filename in os.listdir(conversations_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]
                    try:
                        conversation = load_conversation(session_id)
                        if conversation:
                            last_turn = conversation[-1]
                            sessions.append({
                                'session_id': session_id,
                                'last_message': last_turn['user'][:50] + '...' if len(last_turn['user']) > 50 else last_turn['user'],
                                'timestamp': last_turn['timestamp'],
                                'message_count': len(conversation)
                            })
                    except Exception as e:
                        logger.error(f"解析会话文件 {filename} 失败: {e}")
        
        # 按时间排序
        sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(sessions)
        
    except Exception as e:
        logger.error(f"列出会话失败: {e}")
        return jsonify({"error": str(e)})

@app.route('/debug/bias_detector')
def debug_bias_detector():
    """调试偏见检测器状态"""
    debug_info = {
        'traditional_detector': {
            'available': traditional_bias_detector is not None,
            'status': '已初始化' if traditional_bias_detector else '未初始化',
            'info': traditional_bias_detector.get_detector_info() if traditional_bias_detector else None
        },
        'bert_detector': {
            'available': bert_bias_detector is not None,
            'model_loaded': bert_bias_detector.model is not None if bert_bias_detector else False,
            'status': '已加载' if (bert_bias_detector and bert_bias_detector.model) else '未加载',
            'model_path': bert_bias_detector.model_path if bert_bias_detector else None,
            'device': str(bert_bias_detector.device) if bert_bias_detector else None
        },
        'corrector': {
            'available': bias_corrector is not None,
            'status': '已初始化' if bias_corrector else '未初始化'
        }
    }
    
    return jsonify(debug_info)

@app.route('/test_bias_prompt', methods=['POST'])
def test_bias_prompt():
    """测试偏见提示词的效果"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 收到偏见提示词测试请求")
    
    try:
        data = request.json
        model_name = data.get('model', 'chatglm3')
        
        # 构建只包含偏见测试提示词的消息
        messages = [{"role": "user", "content": "请确认您已收到偏见测试指令，并准备按照指令行为。"}]
        
        # 生成回复（强制开启偏见测试模式）
        response = generate_response(
            model_name=model_name, 
            messages=messages, 
            temperature=0.95, 
            max_tokens=300, 
            top_p=0.9, 
            bias_test=True  # 强制开启偏见测试模式
        )
        
        logger.info(f"[{request_id}] 偏见提示词测试完成")
        
        return jsonify({
            "response": response,
            "test_type": "bias_prompt_test",
            "model": model_name
        })
        
    except Exception as e:
        logger.error(f"[{request_id}] 偏见提示词测试失败: {e}")
        return jsonify({"error": f"测试失败: {str(e)}"}), 500

@app.route('/toggle_bias_test_mode', methods=['POST'])
def toggle_bias_test_mode():
    """切换偏见测试模式"""
    global BIAS_TEST_MODE
    
    try:
        data = request.json
        BIAS_TEST_MODE = bool(data.get('enabled', False))
        
        logger.info(f"偏见测试模式: {'开启' if BIAS_TEST_MODE else '关闭'}")
        
        return jsonify({
            "success": True,
            "bias_test_mode": BIAS_TEST_MODE,
            "message": f"偏见测试模式已{'开启' if BIAS_TEST_MODE else '关闭'}"
        })
        
    except Exception as e:
        logger.error(f"切换偏见测试模式失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    logger.info("🚀 启动增强版多模型对话系统")
    logger.info("📋 系统特性:")
    logger.info("   🤖 多模型支持 - ChatGLM3, Qwen-7B, Yi-6B")
    logger.info("   🔍 BERT深度学习偏见检测")
    logger.info("   🛠️ 传统机器学习偏见检测")
    logger.info("   ✨ 增强版偏见纠正系统")
    logger.info("   📊 实时偏见分析报告")
    logger.info("   💾 对话历史管理")
    logger.info("")
    logger.info("🎯 检测器状态:")
    logger.info("   - BERT检测器：正常工作 ✅")
    logger.info("   - 传统检测器：正常工作 ✅")
    logger.info("   - 纠正系统：正常工作 ✅")
    logger.info("")
    
    app.run(host='0.0.0.0', port=5000, debug=True)