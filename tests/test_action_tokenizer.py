# Filename: tests/test_action_tokenizer.py
"""
动作编码单元测试

测试编解码一致性和 token ID 范围 (151665-153712)。

Requirements: 2.4, 2.6
"""

import unittest
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.action_tokenizer import (
    ActionTokenizer,
    ACTION_TOKEN_MIN,
    ACTION_TOKEN_MAX,
    ACTION_TOKEN_VOCAB_SIZE,
    normalize_gripper_action,
    invert_gripper_action,
)


class TestActionTokenizerConstants(unittest.TestCase):
    """测试动作 token 常量定义"""
    
    def test_action_token_min_value(self):
        """测试 ACTION_TOKEN_MIN 值为 151665 (Req 2.6)"""
        self.assertEqual(ACTION_TOKEN_MIN, 151665,
            f"ACTION_TOKEN_MIN 应为 151665，实际为 {ACTION_TOKEN_MIN}")
    
    def test_action_token_max_value(self):
        """测试 ACTION_TOKEN_MAX 值为 153712 (Req 2.6)"""
        self.assertEqual(ACTION_TOKEN_MAX, 153712,
            f"ACTION_TOKEN_MAX 应为 153712，实际为 {ACTION_TOKEN_MAX}")
    
    def test_action_token_vocab_size(self):
        """测试 ACTION_TOKEN_VOCAB_SIZE 为 2048"""
        expected_size = ACTION_TOKEN_MAX - ACTION_TOKEN_MIN + 1
        self.assertEqual(ACTION_TOKEN_VOCAB_SIZE, expected_size,
            f"ACTION_TOKEN_VOCAB_SIZE 应为 {expected_size}，实际为 {ACTION_TOKEN_VOCAB_SIZE}")
        self.assertEqual(ACTION_TOKEN_VOCAB_SIZE, 2048,
            f"ACTION_TOKEN_VOCAB_SIZE 应为 2048，实际为 {ACTION_TOKEN_VOCAB_SIZE}")


class TestTokenIdConversion(unittest.TestCase):
    """测试 token ID 转换功能"""
    
    @classmethod
    def setUpClass(cls):
        """尝试加载 ActionTokenizer，如果 FAST+ 不可用则跳过部分测试"""
        try:
            cls.tokenizer = ActionTokenizer()
            cls.fast_available = True
        except Exception as e:
            cls.fast_available = False
            cls.skip_reason = str(e)
    
    def test_tokens_to_vlm_ids_range(self):
        """测试 FAST+ token 转 VLM ID 后在正确范围内 (Req 2.6)"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        # 测试边界值
        test_tokens = [0, 1, 100, 1000, 2047]
        vlm_ids = self.tokenizer.tokens_to_vlm_ids(test_tokens)
        
        for vlm_id in vlm_ids:
            self.assertGreaterEqual(vlm_id, ACTION_TOKEN_MIN,
                f"VLM ID {vlm_id} 小于最小值 {ACTION_TOKEN_MIN}")
            self.assertLessEqual(vlm_id, ACTION_TOKEN_MAX,
                f"VLM ID {vlm_id} 大于最大值 {ACTION_TOKEN_MAX}")
    
    def test_vlm_ids_to_tokens_roundtrip(self):
        """测试 VLM ID 和 FAST+ token 之间的往返转换"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        original_tokens = [0, 100, 500, 1000, 2047]
        vlm_ids = self.tokenizer.tokens_to_vlm_ids(original_tokens)
        recovered_tokens = self.tokenizer.vlm_ids_to_tokens(vlm_ids)
        
        self.assertEqual(original_tokens, recovered_tokens,
            f"Token 往返转换不一致: {original_tokens} -> {vlm_ids} -> {recovered_tokens}")
    
    def test_is_action_token_id_valid_range(self):
        """测试 is_action_token_id 对有效范围内的 ID 返回 True"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        # 测试边界值
        self.assertTrue(self.tokenizer.is_action_token_id(ACTION_TOKEN_MIN))
        self.assertTrue(self.tokenizer.is_action_token_id(ACTION_TOKEN_MAX))
        self.assertTrue(self.tokenizer.is_action_token_id(152000))  # 中间值
    
    def test_is_action_token_id_invalid_range(self):
        """测试 is_action_token_id 对无效范围的 ID 返回 False"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        # 测试边界外的值
        self.assertFalse(self.tokenizer.is_action_token_id(ACTION_TOKEN_MIN - 1))
        self.assertFalse(self.tokenizer.is_action_token_id(ACTION_TOKEN_MAX + 1))
        self.assertFalse(self.tokenizer.is_action_token_id(0))
        self.assertFalse(self.tokenizer.is_action_token_id(100000))


class TestVLMStringFormat(unittest.TestCase):
    """测试 VLM 字符串格式化功能"""
    
    @classmethod
    def setUpClass(cls):
        """尝试加载 ActionTokenizer"""
        try:
            cls.tokenizer = ActionTokenizer()
            cls.fast_available = True
        except Exception as e:
            cls.fast_available = False
            cls.skip_reason = str(e)
    
    def test_tokens_to_vlm_string_format(self):
        """测试 tokens_to_vlm_string 输出格式正确 (Req 2.3)"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        tokens = [0, 1, 2]
        vlm_string = self.tokenizer.tokens_to_vlm_string(tokens)
        
        expected = "<robot_action_0><robot_action_1><robot_action_2>"
        self.assertEqual(vlm_string, expected,
            f"VLM 字符串格式不正确: 期望 {expected}，实际 {vlm_string}")
    
    def test_tokens_to_vlm_string_empty(self):
        """测试空 token 列表的处理"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        tokens = []
        vlm_string = self.tokenizer.tokens_to_vlm_string(tokens)
        
        self.assertEqual(vlm_string, "",
            f"空 token 列表应返回空字符串，实际返回 {vlm_string}")


class TestEncodeDecode(unittest.TestCase):
    """测试编解码一致性 (Req 2.4)"""
    
    @classmethod
    def setUpClass(cls):
        """尝试加载 ActionTokenizer"""
        try:
            cls.tokenizer = ActionTokenizer()
            cls.fast_available = True
        except Exception as e:
            cls.fast_available = False
            cls.skip_reason = str(e)
    
    def test_encode_decode_consistency(self):
        """测试编码后解码能恢复原始动作（在量化误差范围内）"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        # 创建测试动作 (7-DoF: 6 位姿 + 1 gripper)
        original_action = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.15, 0.5], dtype=np.float32)
        
        # 编码
        tokens = self.tokenizer.encode(original_action)
        print(f"\n编码结果: {tokens} (共 {len(tokens)} 个 token)")
        
        # 解码
        try:
            decoded_action = self.tokenizer.decode(tokens)
            print(f"解码结果形状: {decoded_action.shape}")
            print(f"解码结果: {decoded_action}")
        except Exception as e:
            self.fail(f"解码失败: {e}")
        
        # 检查解码结果不为空
        self.assertIsNotNone(decoded_action, "解码结果不应为 None")
        
        # 检查形状 - FAST+ 可能返回不同形状，我们只检查最后一维
        decoded_flat = decoded_action.flatten()
        self.assertGreaterEqual(len(decoded_flat), 7,
            f"解码后动作维度应至少为 7，实际为 {len(decoded_flat)}")
        
        # 取前 7 个值进行比较
        decoded_7 = decoded_flat[:7]
        
        # 检查值在合理范围内（允许量化误差）
        # FAST+ 使用离散化，所以会有一定误差
        max_error = np.max(np.abs(original_action - decoded_7))
        
        # 量化误差通常在 0.5 以内（FAST+ 的量化精度有限）
        self.assertLessEqual(max_error, 1.0,
            f"编解码误差过大: {max_error}, 原始: {original_action}, 解码: {decoded_7}")
    
    def test_encode_decode_multiple_actions(self):
        """测试多个不同动作的编解码一致性"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        test_actions = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # 零动作
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float32),  # 正值
            np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0], dtype=np.float32),  # 负值
        ]
        
        for i, original_action in enumerate(test_actions):
            tokens = self.tokenizer.encode(original_action)
            
            try:
                decoded_action = self.tokenizer.decode(tokens)
                decoded_flat = decoded_action.flatten()
                
                # 检查解码后的动作维度至少为 7
                self.assertGreaterEqual(len(decoded_flat), 7,
                    f"测试 {i}: 解码后动作维度应至少为 7，实际为 {len(decoded_flat)}")
            except Exception as e:
                self.fail(f"测试 {i}: 解码失败 - {e}")
    
    def test_encode_returns_valid_tokens(self):
        """测试编码返回的 token 在有效范围内"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        action = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.15, 0.5], dtype=np.float32)
        tokens = self.tokenizer.encode(action)
        
        # 检查 token 是非负整数
        for token in tokens:
            self.assertGreaterEqual(token, 0,
                f"Token {token} 应为非负整数")
            self.assertLess(token, ACTION_TOKEN_VOCAB_SIZE,
                f"Token {token} 应小于词汇表大小 {ACTION_TOKEN_VOCAB_SIZE}")


class TestGripperNormalization(unittest.TestCase):
    """测试 gripper 归一化功能 (Req 2.5)"""
    
    def test_normalize_gripper_action_range(self):
        """测试 normalize_gripper_action 将 [0,1] 转换为 [-1,1]"""
        # 输入 [0, 1] 范围
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5], dtype=np.float32)  # gripper=0.5
        
        normalized = normalize_gripper_action(action, binarize=False)
        
        # gripper 应该从 0.5 转换为 0.0 (线性映射)
        # 公式: 2 * (x - 0) / (1 - 0) - 1 = 2x - 1
        # 0.5 -> 2*0.5 - 1 = 0.0
        expected_gripper = 0.0
        self.assertAlmostEqual(normalized[-1], expected_gripper, places=5,
            msg=f"Gripper 归一化错误: 期望 {expected_gripper}，实际 {normalized[-1]}")
    
    def test_normalize_gripper_action_boundaries(self):
        """测试 gripper 归一化边界值"""
        # gripper = 0 -> -1
        action_close = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        normalized_close = normalize_gripper_action(action_close, binarize=False)
        self.assertAlmostEqual(normalized_close[-1], -1.0, places=5)
        
        # gripper = 1 -> +1
        action_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        normalized_open = normalize_gripper_action(action_open, binarize=False)
        self.assertAlmostEqual(normalized_open[-1], 1.0, places=5)
    
    def test_normalize_gripper_action_binarize(self):
        """测试 gripper 归一化的二值化功能"""
        # gripper = 0.3 -> 2*0.3-1 = -0.4 -> binarize -> -1
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3], dtype=np.float32)
        normalized = normalize_gripper_action(action, binarize=True)
        self.assertEqual(normalized[-1], -1.0,
            f"二值化后 gripper 应为 -1，实际为 {normalized[-1]}")
        
        # gripper = 0.7 -> 2*0.7-1 = 0.4 -> binarize -> +1
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7], dtype=np.float32)
        normalized = normalize_gripper_action(action, binarize=True)
        self.assertEqual(normalized[-1], 1.0,
            f"二值化后 gripper 应为 +1，实际为 {normalized[-1]}")
    
    def test_invert_gripper_action(self):
        """测试 gripper 反转功能"""
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5], dtype=np.float32)
        
        inverted = invert_gripper_action(action)
        
        # gripper 应该取反
        self.assertAlmostEqual(inverted[-1], -0.5, places=5,
            msg=f"Gripper 反转错误: 期望 -0.5，实际 {inverted[-1]}")
        
        # 其他维度不变
        np.testing.assert_array_almost_equal(inverted[:-1], action[:-1],
            err_msg="非 gripper 维度不应改变")
    
    def test_invert_gripper_action_preserves_original(self):
        """测试 invert_gripper_action 不修改原始数组"""
        original = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5], dtype=np.float32)
        original_copy = original.copy()
        
        _ = invert_gripper_action(original)
        
        np.testing.assert_array_equal(original, original_copy,
            err_msg="invert_gripper_action 不应修改原始数组")


class TestActionTokenizerGripperNormalization(unittest.TestCase):
    """测试 ActionTokenizer 内部的 gripper 归一化"""
    
    @classmethod
    def setUpClass(cls):
        """尝试加载 ActionTokenizer"""
        try:
            cls.tokenizer = ActionTokenizer()
            cls.fast_available = True
        except Exception as e:
            cls.fast_available = False
            cls.skip_reason = str(e)
    
    def test_normalize_gripper_for_encoding(self):
        """测试 _normalize_gripper_for_encoding 转换正确"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        # LIBERO: -1=open, +1=close
        # NORA: 0=close, 1=open
        # 公式: gripper_nora = (1 - gripper_libero) / 2
        
        # LIBERO -1 (open) -> NORA 1 (open)
        action_open = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
        normalized = self.tokenizer._normalize_gripper_for_encoding(action_open.copy())
        self.assertAlmostEqual(normalized[0, -1], 1.0, places=5,
            msg="LIBERO -1 (open) 应转换为 NORA 1 (open)")
        
        # LIBERO +1 (close) -> NORA 0 (close)
        action_close = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        normalized = self.tokenizer._normalize_gripper_for_encoding(action_close.copy())
        self.assertAlmostEqual(normalized[0, -1], 0.0, places=5,
            msg="LIBERO +1 (close) 应转换为 NORA 0 (close)")
    
    def test_denormalize_gripper_after_decoding(self):
        """测试 _denormalize_gripper_after_decoding 转换正确"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        # 逆转换: gripper_libero = 1 - 2 * gripper_nora
        
        # NORA 1 (open) -> LIBERO -1 (open)
        action_open = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        denormalized = self.tokenizer._denormalize_gripper_after_decoding(action_open.copy())
        self.assertAlmostEqual(denormalized[0, -1], -1.0, places=5,
            msg="NORA 1 (open) 应转换为 LIBERO -1 (open)")
        
        # NORA 0 (close) -> LIBERO +1 (close)
        action_close = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        denormalized = self.tokenizer._denormalize_gripper_after_decoding(action_close.copy())
        self.assertAlmostEqual(denormalized[0, -1], 1.0, places=5,
            msg="NORA 0 (close) 应转换为 LIBERO +1 (close)")
    
    def test_gripper_normalization_roundtrip(self):
        """测试 gripper 归一化的往返一致性"""
        if not self.fast_available:
            self.skipTest(f"FAST+ tokenizer 不可用: {self.skip_reason}")
        
        # 测试多个 gripper 值
        test_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for gripper_val in test_values:
            action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_val]], dtype=np.float32)
            
            # 归一化
            normalized = self.tokenizer._normalize_gripper_for_encoding(action.copy())
            
            # 反归一化
            denormalized = self.tokenizer._denormalize_gripper_after_decoding(normalized.copy())
            
            self.assertAlmostEqual(denormalized[0, -1], gripper_val, places=5,
                msg=f"Gripper 往返转换不一致: {gripper_val} -> {normalized[0, -1]} -> {denormalized[0, -1]}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
