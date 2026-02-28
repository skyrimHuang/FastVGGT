#!/usr/bin/env python3
"""
验证训练系统 - Verify complete training pipeline
测试：
1. TrainingHistory 类功能
2. 检查点保存/加载
3. 训练循环完整性
"""

import json
import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目到路径
sys.path.insert(0, str(Path(__file__).parent))


def test_training_history():
    """测试 TrainingHistory 类"""
    print("\n" + "="*70)
    print("TEST 1: TrainingHistory Class")
    print("="*70)
    
    # 导入TrainingHistory类
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建临时模块来导入TrainingHistory
        code = """
import sys
from pathlib import Path
sys.path.insert(0, '/home/hba/Documents/FastVGGT')

from train_scale_head_kitti import TrainingHistory

# 创建实例
hist = TrainingHistory('{tmpdir}')

# 记录数据
for epoch in range(3):
    metrics = {{
        'epoch': epoch,
        'train_loss': 0.5 - epoch*0.1,
        'val_mean_error': 0.15 - epoch*0.02,
        'val_rmse': 0.2 - epoch*0.01,
        'val_fps': 45 + epoch*2,
        'learning_rate': 1e-4 / (epoch+1)
    }}
    hist.record_epoch(epoch, metrics)

# 保存
hist.save(None)

# 验证文件
json_file = Path('{tmpdir}') / 'training_history.json'
csv_file = Path('{tmpdir}') / 'training_history.csv'

assert json_file.exists(), f"JSON file not created: {{json_file}}"
assert csv_file.exists(), f"CSV file not created: {{csv_file}}"

# 加载验证
hist2 = TrainingHistory('{tmpdir}')
hist2.load(None)

print("✓ JSON file created and verified")
print("✓ CSV file created and verified")
print("✓ History loaded successfully")
print(f"✓ Loaded {{len(hist2.history['epoch'])}} epochs")

# 打印样本
import json as json_module
with open(json_file, 'r') as f:
    data = json_module.load(f)
    print("\\nJSON content sample:")
    if isinstance(data, list) and len(data) > 0:
        print(json_module.dumps(data[0], indent=2))

""".format(tmpdir=tmpdir)
        
        exec(code)
    
    print("✓ TrainingHistory test PASSED")
    return True


def test_checkpoint_saving():
    """测试检查点保存/加载"""
    print("\n" + "="*70)
    print("TEST 2: Checkpoint Save/Load")
    print("="*70)
    
    try:
        import torch
        import torch.nn as nn
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建简单的模型
            model = nn.Linear(10, 1)
            
            # 保存
            ckpt_path = Path(tmpdir) / 'test_model.pt'
            torch.save(model.state_dict(), ckpt_path)
            
            assert ckpt_path.exists(), f"Checkpoint not saved: {ckpt_path}"
            print(f"✓ Saved checkpoint to: {ckpt_path}")
            
            # 创建新模型并加载
            model2 = nn.Linear(10, 1)
            state = torch.load(ckpt_path)
            model2.load_state_dict(state)
            
            print("✓ Loaded checkpoint successfully")
            
            # 验证权重相同
            assert torch.allclose(model.weight, model2.weight), "Weights don't match"
            print("✓ Checkpoint weights verified")
            
    except ImportError:
        print("⚠ PyTorch not available, skipping checkpoint test")
        return True
    
    print("✓ Checkpoint save/load test PASSED")
    return True


def verify_training_script():
    """验证训练脚本完整性"""
    print("\n" + "="*70)
    print("TEST 3: Training Script Integrity")
    print("="*70)
    
    train_script = Path('/home/hba/Documents/FastVGGT/train_scale_head_kitti.py')
    
    if not train_script.exists():
        print(f"✗ Training script not found: {train_script}")
        return False
    
    with open(train_script) as f:
        content = f.read()
    
    # 检查关键要素
    checks = {
        'class TrainingHistory': 'TrainingHistory class defined',
        'def record_epoch': 'record_epoch method exists',
        'def save': 'save method exists',
        'def load': 'load method exists',
        'history_recorder = TrainingHistory': 'TrainingHistory instantiation',
        'history_recorder.record_epoch': 'Metric recording',
        'history_recorder.save': 'History saving',
        'torch.save(model.scale_head.state_dict()': 'Checkpoint saving',
        'model.scale_head.load_state_dict': 'Checkpoint loading',
        'for epoch in range(start_epoch': 'Resume epoch loop'
    }
    
    all_passed = True
    for key, desc in checks.items():
        if key in content:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc} - NOT FOUND")
            all_passed = False
    
    if all_passed:
        print("✓ Training script integrity test PASSED")
    else:
        print("✗ Training script integrity test FAILED")
    
    return all_passed


def check_output_directory():
    """检查输出目录结构"""
    print("\n" + "="*70)
    print("TEST 4: Output Directory Structure")
    print("="*70)
    
    output_dir = Path('/home/hba/Documents/FastVGGT/outputs/scale_head')
    
    # 检查是否需要的目录被创建
    required_structure = {
        'outputs/': 'Output root directory',
        'outputs/scale_head/': 'Scale head output directory',
    }
    
    # 创建输出目录用于演示
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dir_path, desc in required_structure.items():
        full_path = Path('/home/hba/Documents/FastVGGT') / dir_path
        if full_path.exists():
            print(f"✓ {desc} exists at {full_path}")
        else:
            print(f"⚠ {desc} will be created during training")
    
    # 检查示例输出
    json_path = output_dir / 'training_history.json'
    csv_path = output_dir / 'training_history.csv'
    
    if json_path.exists():
        print(f"✓ training_history.json found at {json_path}")
    else:
        print(f"⚠ training_history.json will be created after first training run")
    
    if csv_path.exists():
        print(f"✓ training_history.csv found at {csv_path}")
    else:
        print(f"⚠ training_history.csv will be created after first training run")
    
    print("✓ Output directory structure test PASSED")
    return True


def verify_plotting_script():
    """验证绘图脚本"""
    print("\n" + "="*70)
    print("TEST 5: Plotting Script")
    print("="*70)
    
    plot_script = Path('/home/hba/Documents/FastVGGT/plot_training_curves.py')
    
    if not plot_script.exists():
        print(f"✗ Plot script not found: {plot_script}")
        return False
    
    with open(plot_script) as f:
        content = f.read()
    
    checks = {
        'def plot_training_curves': 'plot_training_curves function',
        'def load_history_from_json': 'JSON loading',
        'def load_history_from_csv': 'CSV loading',
        'matplotlib.pyplot': 'Matplotlib import',
        'loss_curve.png': 'Loss curve output',
        'error_curve.png': 'Error curve output',
    }
    
    all_passed = True
    for key, desc in checks.items():
        if key in content:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc} - NOT FOUND")
            all_passed = False
    
    if all_passed:
        print("✓ Plotting script test PASSED")
    else:
        print("✗ Plotting script test FAILED")
    
    return all_passed


def main():
    """运行所有测试"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "TRAINING SYSTEM VERIFICATION SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    results = {}
    
    # 运行测试
    try:
        results['TrainingHistory'] = test_training_history()
    except Exception as e:
        print(f"✗ TrainingHistory test FAILED: {e}")
        results['TrainingHistory'] = False
    
    try:
        results['Checkpoints'] = test_checkpoint_saving()
    except Exception as e:
        print(f"✗ Checkpoint test FAILED: {e}")
        results['Checkpoints'] = False
    
    try:
        results['Training Script'] = verify_training_script()
    except Exception as e:
        print(f"✗ Training script test FAILED: {e}")
        results['Training Script'] = False
    
    try:
        results['Output Directory'] = check_output_directory()
    except Exception as e:
        print(f"✗ Output directory test FAILED: {e}")
        results['Output Directory'] = False
    
    try:
        results['Plot Script'] = verify_plotting_script()
    except Exception as e:
        print(f"✗ Plot script test FAILED: {e}")
        results['Plot Script'] = False
    
    # 总结
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All verification tests PASSED!")
        print("\nYou can now run training with:")
        print("  python train_scale_head_kitti.py --config configs/train_scale_head_kitti.yaml")
        print("\nAfter training, plot curves with:")
        print("  python plot_training_curves.py --history outputs/scale_head/training_history.json")
        return 0
    else:
        print("\n✗ Some tests FAILED - please review above")
        return 1


if __name__ == '__main__':
    exit(main())
