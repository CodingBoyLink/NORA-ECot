# Filename: tests/test_data_splitter.py
"""
数据加载单元测试

测试数据划分正确性和每个任务在 train/test 中的覆盖。

Requirements: 1.2, 1.3
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.data_splitter import LiberoDataSplitter, SplitInfo


class TestLiberoDataSplitter(unittest.TestCase):
    """测试 LiberoDataSplitter 数据划分功能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.data_dir = "./data"
        cls.subset = "object"  # 使用 libero_object 子集
        cls.train_ratio = 0.95
        cls.seed = 42
        
        # 检查数据目录是否存在
        subset_dir = os.path.join(cls.data_dir, f"libero_{cls.subset}")
        if not os.path.exists(subset_dir):
            raise unittest.SkipTest(f"数据目录不存在: {subset_dir}")
        
        cls.splitter = LiberoDataSplitter(
            data_dir=cls.data_dir,
            subset=cls.subset,
            train_ratio=cls.train_ratio,
            seed=cls.seed
        )
        cls.split_info = cls.splitter.split()
    
    def test_split_ratio_approximately_correct(self):
        """测试数据划分比例接近 95/5"""
        train_count, test_count = self.splitter.get_total_trajectories()
        total = train_count + test_count
        
        actual_train_ratio = train_count / total
        
        # 允许 5% 的误差（因为要确保每个任务都有覆盖）
        self.assertGreater(actual_train_ratio, 0.85, 
            f"训练集比例 {actual_train_ratio:.2%} 过低")
        self.assertLess(actual_train_ratio, 1.0,
            f"训练集比例 {actual_train_ratio:.2%} 过高，测试集为空")
        
        print(f"\n划分比例: train={train_count} ({actual_train_ratio:.1%}), "
              f"test={test_count} ({1-actual_train_ratio:.1%})")
    
    def test_every_task_has_train_coverage(self):
        """测试每个任务在训练集中都有覆盖"""
        task_dist = self.split_info.task_distribution
        
        tasks_without_train = []
        for task_name, dist in task_dist.items():
            if dist['train'] == 0:
                tasks_without_train.append(task_name)
        
        self.assertEqual(len(tasks_without_train), 0,
            f"以下任务在训练集中没有覆盖: {tasks_without_train}")
        
        print(f"\n所有 {len(task_dist)} 个任务在训练集中都有覆盖")
    
    def test_every_task_has_test_coverage(self):
        """测试每个任务在测试集中都有覆盖"""
        task_dist = self.split_info.task_distribution
        
        tasks_without_test = []
        for task_name, dist in task_dist.items():
            if dist['test'] == 0:
                tasks_without_test.append(task_name)
        
        self.assertEqual(len(tasks_without_test), 0,
            f"以下任务在测试集中没有覆盖: {tasks_without_test}")
        
        print(f"\n所有 {len(task_dist)} 个任务在测试集中都有覆盖")
    
    def test_split_reproducibility(self):
        """测试使用相同种子的划分可复现性"""
        splitter2 = LiberoDataSplitter(
            data_dir=self.data_dir,
            subset=self.subset,
            train_ratio=self.train_ratio,
            seed=self.seed
        )
        split_info2 = splitter2.split()
        
        # 比较训练集轨迹索引
        for file_path in self.split_info.train_trajectories:
            self.assertEqual(
                self.split_info.train_trajectories[file_path],
                split_info2.train_trajectories[file_path],
                f"文件 {file_path} 的训练集划分不一致"
            )
        
        print("\n使用相同种子的划分结果一致")
    
    def test_no_overlap_between_train_and_test(self):
        """测试训练集和测试集没有重叠（除非只有1条轨迹）"""
        for file_path in self.split_info.train_files:
            train_indices = set(self.split_info.train_trajectories[file_path])
            test_indices = set(self.split_info.test_trajectories[file_path])
            
            overlap = train_indices & test_indices
            
            # 如果有重叠，应该是因为该任务只有1条轨迹
            if overlap:
                task_dist = self.split_info.task_distribution
                task_name = os.path.basename(file_path).replace('_demo.hdf5', '')
                total = task_dist.get(task_name, {}).get('total', 0)
                self.assertEqual(total, 1,
                    f"文件 {file_path} 有重叠但轨迹数不为1")
        
        print("\n训练集和测试集无非法重叠")


if __name__ == '__main__':
    unittest.main(verbosity=2)
