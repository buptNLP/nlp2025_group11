#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChatGLM兼容性补丁模块
专门用于修复ChatGLMTokenizer的padding_side参数问题
"""

import logging
import types

logger = logging.getLogger(__name__)

def apply_chatglm_tokenizer_patch():
    """应用ChatGLM tokenizer的兼容性补丁"""
    
    try:
        logger.info("🔧 开始应用ChatGLM全面兼容性补丁...")
        
        # 1. 首先修复PreTrainedTokenizerBase
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        
        if not hasattr(PreTrainedTokenizerBase, '_original_pad_method'):
            PreTrainedTokenizerBase._original_pad_method = PreTrainedTokenizerBase._pad
            
            def safe_pad(self, encoded_inputs, *args, **kwargs):
                # 移除不支持的参数
                kwargs.pop('padding_side', None)
                kwargs.pop('pad_to_multiple_of', None)
                return PreTrainedTokenizerBase._original_pad_method(self, encoded_inputs, *args, **kwargs)
            
            PreTrainedTokenizerBase._pad = safe_pad
            logger.info("✅ PreTrainedTokenizerBase._pad 补丁应用成功")
        
        # 2. 修复AutoTokenizer的from_pretrained方法
        from transformers import AutoTokenizer
        
        if not hasattr(AutoTokenizer, '_original_from_pretrained'):
            AutoTokenizer._original_from_pretrained = AutoTokenizer.from_pretrained
            
            @classmethod
            def patched_from_pretrained(cls, *args, **kwargs):
                tokenizer = AutoTokenizer._original_from_pretrained(*args, **kwargs)
                
                # 检查是否是ChatGLMTokenizer并应用补丁
                if hasattr(tokenizer, '_pad') and 'ChatGLM' in tokenizer.__class__.__name__:
                    logger.info(f"🎯 检测到ChatGLMTokenizer: {tokenizer.__class__.__name__}")
                    
                    # 保存原始方法
                    original_pad = tokenizer._pad
                    
                    def safe_tokenizer_pad(encoded_inputs, *args, **kwargs):
                        # 移除不支持的参数
                        kwargs.pop('padding_side', None)
                        kwargs.pop('pad_to_multiple_of', None)
                        return original_pad(encoded_inputs, *args, **kwargs)
                    
                    # 绑定新方法到实例
                    tokenizer._pad = types.MethodType(lambda self, *args, **kwargs: safe_tokenizer_pad(*args, **kwargs), tokenizer)
                    logger.info("✅ ChatGLMTokenizer实例补丁应用成功")
                
                return tokenizer
            
            AutoTokenizer.from_pretrained = patched_from_pretrained
            logger.info("✅ AutoTokenizer.from_pretrained 补丁应用成功")
        
        # 3. 修复ChatGLM模型的_extract_past_from_model_output方法
        apply_chatglm_model_patch()
        
        logger.info("🎉 所有ChatGLM兼容性补丁应用完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ ChatGLM兼容性补丁应用失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def apply_chatglm_model_patch():
    """应用ChatGLM模型的兼容性补丁"""
    try:
        logger.info("🔧 应用ChatGLM模型兼容性补丁...")
        from transformers import AutoModelForCausalLM
        
        # 修复AutoModelForCausalLM的from_pretrained方法
        if not hasattr(AutoModelForCausalLM, '_original_model_from_pretrained'):
            AutoModelForCausalLM._original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
            
            @classmethod
            def patched_model_from_pretrained(cls, *args, **kwargs):
                model = AutoModelForCausalLM._original_model_from_pretrained(*args, **kwargs)
                
                # 检查是否是ChatGLM模型并应用补丁
                if 'ChatGLM' in model.__class__.__name__:
                    logger.info(f"🎯 检测到ChatGLM模型: {model.__class__.__name__}")
                    
                    # 添加缺失的_extract_past_from_model_output方法
                    if not hasattr(model, '_extract_past_from_model_output'):
                        def _extract_past_from_model_output(self, outputs, standardize_cache_format=False):
                            """从模型输出中提取past_key_values"""
                            if hasattr(outputs, 'past_key_values'):
                                return outputs.past_key_values
                            elif isinstance(outputs, tuple) and len(outputs) > 1:
                                # 假设past_key_values是第二个元素
                                return outputs[1] if outputs[1] is not None else None
                            else:
                                return None
                        
                        # 绑定方法到模型实例
                        model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, model)
                        logger.info("✅ ChatGLM模型_extract_past_from_model_output方法已添加")
                
                return model
            
            AutoModelForCausalLM.from_pretrained = patched_model_from_pretrained
            logger.info("✅ AutoModelForCausalLM.from_pretrained 补丁应用成功")
        
        logger.info("✅ ChatGLM模型兼容性补丁应用完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ ChatGLM模型补丁应用失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def patch_existing_tokenizer(tokenizer):
    """为已加载的tokenizer应用补丁"""
    try:
        if hasattr(tokenizer, '_pad') and 'ChatGLM' in tokenizer.__class__.__name__:
            logger.info(f"🎯 为现有tokenizer应用补丁: {tokenizer.__class__.__name__}")
            
            # 保存原始方法
            original_pad = tokenizer._pad
            
            def safe_existing_pad(encoded_inputs, *args, **kwargs):
                # 移除不支持的参数
                kwargs.pop('padding_side', None)
                kwargs.pop('pad_to_multiple_of', None)
                return original_pad(encoded_inputs, *args, **kwargs)
            
            # 绑定新方法
            tokenizer._pad = types.MethodType(lambda self, *args, **kwargs: safe_existing_pad(*args, **kwargs), tokenizer)
            logger.info("✅ 现有tokenizer补丁应用成功")
            return True
    except Exception as e:
        logger.error(f"❌ 现有tokenizer补丁应用失败: {e}")
    
    return False

def patch_existing_model(model):
    """为已加载的模型应用补丁"""
    try:
        if 'ChatGLM' in model.__class__.__name__:
            logger.info(f"🎯 为现有模型应用补丁: {model.__class__.__name__}")
            
            # 添加缺失的_extract_past_from_model_output方法
            if not hasattr(model, '_extract_past_from_model_output'):
                def _extract_past_from_model_output(self, outputs, standardize_cache_format=False):
                    """从模型输出中提取past_key_values"""
                    if hasattr(outputs, 'past_key_values'):
                        return outputs.past_key_values
                    elif isinstance(outputs, tuple) and len(outputs) > 1:
                        # 假设past_key_values是第二个元素
                        return outputs[1] if outputs[1] is not None else None
                    else:
                        return None
                
                # 绑定方法到模型实例
                model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, model)
                logger.info("✅ 现有模型_extract_past_from_model_output方法已添加")
                return True
    except Exception as e:
        logger.error(f"❌ 现有模型补丁应用失败: {e}")
    
    return False

# 自动应用补丁（当模块被导入时）
if __name__ != "__main__":
    apply_chatglm_tokenizer_patch() 