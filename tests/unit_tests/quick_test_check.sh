#!/bin/bash
# 单元测试快速检查脚本

echo "========================================="
echo "VGGT KITTI Scale Head 单元测试检查"
echo "========================================="
echo ""

cd /home/hba/Documents/FastVGGT

echo "[1/5] 测试 KITTI Calibration..."
conda run -n fastvggt python tests/unit_tests/test_01_kitti_calib.py > /tmp/test01.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ test_01_kitti_calib.py - 通过"
else
    echo "❌ test_01_kitti_calib.py - 失败"
    tail -5 /tmp/test01.log
fi

echo ""
echo "[2/5] 测试 Scale Head..."
conda run -n fastvggt python tests/unit_tests/test_02_scale_head.py > /tmp/test02.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ test_02_scale_head.py - 通过"
else
    echo "❌ test_02_scale_head.py - 失败"
    tail -5 /tmp/test02.log
fi

echo ""
echo "[3/5] 测试 VGGT Integration..."
conda run -n fastvggt python tests/unit_tests/test_03_vggt_integration.py > /tmp/test03.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ test_03_vggt_integration.py - 通过"
else
    echo "❌ test_03_vggt_integration.py - 失败"
    echo "  (查看详情: cat /tmp/test03.log)"
fi

echo ""
echo "[4/5] 测试 KITTI Dataset..."
conda run -n fastvggt python tests/unit_tests/test_04_kitti_dataset.py > /tmp/test04.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ test_04_kitti_dataset.py - 通过"
else
    echo "❌ test_04_kitti_dataset.py - 失败"
    echo "  (已知问题: API参数不匹配)"
fi

echo ""
echo "[5/5] 测试 Training Components..."
conda run -n fastvggt python tests/unit_tests/test_05_training_components.py > /tmp/test05.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ test_05_training_components.py - 通过"
else
    echo "❌ test_05_training_components.py - 失败"
    tail -10 /tmp/test05.log
fi

echo ""
echo "========================================="
echo "测试总结"
echo "========================================="
grep "通过:" /tmp/test*.log | tail -5
echo ""
echo "详细日志:"
echo "  test_01: /tmp/test01.log"
echo "  test_02: /tmp/test02.log"
echo "  test_03: /tmp/test03.log (可能需要检查dtype问题)"
echo "  test_04: /tmp/test04.log (需要修复API)"
echo "  test_05: /tmp/test05.log"
