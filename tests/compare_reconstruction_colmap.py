"""
Token分割方法的点云重建质量对比脚本
=======================================================
比较Grid-based vs Norm-guided 两种方法的三维重建结果
生成COLMAP格式的重建和可视化的点云对比
"""

import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np

try:
    import pycolmap
    COLMAP_AVAILABLE = True
except ImportError:
    COLMAP_AVAILABLE = False
    print("⚠️  pycolmap not available - point cloud export disabled")
    pycolmap = None

try:
    import trimesh
    import open3d as o3d
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("⚠️  trimesh/open3d not available")


def compute_chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> Tuple[float, float, float]:
    """
    计算Chamfer距离
    
    Args:
        points1: 点云1 [N, 3]
        points2: 点云2 [M, 3]
    
    Returns:
        (单向距离, 双向平均距离)
    """
    from scipy.spatial.distance import cdist
    
    # 中心对齐
    p1_center = points1 - points1.mean(axis=0)
    p2_center = points2 - points2.mean(axis=0)
    
    # 计算距离矩阵
    distances = cdist(p1_center, p2_center)
    
    # 单向距离
    d1_to_2 = distances.min(axis=1).mean()
    d2_to_1 = distances.min(axis=0).mean()
    
    # 双向平均
    chamfer_avg = (d1_to_2 + d2_to_1) / 2.0
    
    return d1_to_2, d2_to_1, chamfer_avg


def load_colmap_reconstruction(colmap_path: Path):
    """加载COLMAP重建结果"""
    if not COLMAP_AVAILABLE:
        print("❌ pycolmap不可用")
        return None
    
    try:
        recon = pycolmap.Reconstruction(str(colmap_path))
        return recon
    except Exception as e:
        print(f"❌ 加载COLMAP重建失败: {e}")
        return None


def extract_pointcloud(reconstruction) -> Tuple[np.ndarray, np.ndarray]:
    """
    从COLMAP重建提取点云和颜色
    
    Returns:
        (点云坐标, 点的颜色)
    """
    points_3d = []
    colors = []
    
    for point_id, point3d in reconstruction.points3D.items():
        points_3d.append(point3d.xyz)
        colors.append(point3d.color)
    
    return np.array(points_3d), np.array(colors, dtype=np.uint8)


def compare_reconstructions(baseline_colmap: Path, 
                           proposed_colmap: Path,
                           output_dir: Path) -> dict:
    """
    对比两个COLMAP重建结果
    
    Args:
        baseline_colmap: 基线方法的COLMAP稀疏重建目录
        proposed_colmap: 本文方法的COLMAP稀疏重建目录
        output_dir: 输出目录
    
    Returns:
        对比结果字典
    """
    print("\n" + "="*80)
    print("📊 加载和对比COLMAP重建...")
    print("="*80)
    
    # 加载重建
    print(f"\n📂 基线方法: {baseline_colmap}")
    baseline_recon = load_colmap_reconstruction(baseline_colmap)
    if not baseline_recon:
        return None
    
    print(f"📂 本文方法: {proposed_colmap}")
    proposed_recon = load_colmap_reconstruction(proposed_colmap)
    if not proposed_recon:
        return None
    
    # 统计信息
    print(f"\n📈 重建统计:")
    print(f"  基线方法:")
    print(f"    - 3D点数: {len(baseline_recon.points3D)}")
    print(f"    - 图像数: {len(baseline_recon.images)}")
    print(f"    - 摄像机数: {len(baseline_recon.cameras)}")
    
    print(f"  本文方法:")
    print(f"    - 3D点数: {len(proposed_recon.points3D)}")
    print(f"    - 图像数: {len(proposed_recon.images)}")
    print(f"    - 摄像机数: {len(proposed_recon.cameras)}")
    
    # 提取点云
    baseline_points, baseline_colors = extract_pointcloud(baseline_recon)
    proposed_points, proposed_colors = extract_pointcloud(proposed_recon)
    
    results = {
        "baseline_num_points": len(baseline_points),
        "proposed_num_points": len(proposed_points),
    }
    
    # 计算Chamfer距离
    if len(baseline_points) > 0 and len(proposed_points) > 0:
        print(f"\n📏 计算重建质量指标...")
        
        d1_to_2, d2_to_1, chamfer = compute_chamfer_distance(baseline_points, proposed_points)
        
        print(f"  Chamfer距离:")
        print(f"    - 基线→本文: {d1_to_2:.6f}")
        print(f"    - 本文→基线: {d2_to_1:.6f}")
        print(f"    - 平均: {chamfer:.6f}")
        
        results["chamfer_baseline_to_proposed"] = float(d1_to_2)
        results["chamfer_proposed_to_baseline"] = float(d2_to_1)
        results["chamfer_average"] = float(chamfer)
        
        # 估计改善百分比
        improvement_pct = ((d1_to_2 - d2_to_1) / d1_to_2) * 100 if d1_to_2 > 0 else 0
        results["improvement_percentage"] = float(improvement_pct)
        print(f"    - 相对改善: {improvement_pct:.2f}%")
    
    # 导出PLY点云
    if TRIMESH_AVAILABLE:
        print(f"\n💾 导出点云文件...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 基线方法点云
        baseline_ply = output_dir / "baseline_grid_based.ply"
        try:
            trimesh.PointCloud(baseline_points, colors=baseline_colors).export(str(baseline_ply))
            print(f"  ✓ {baseline_ply.name}")
        except Exception as e:
            print(f"  ❌ 导出基线点云失败: {e}")
        
        # 本文方法点云
        proposed_ply = output_dir / "proposed_norm_guided.ply"
        try:
            trimesh.PointCloud(proposed_points, colors=proposed_colors).export(str(proposed_ply))
            print(f"  ✓ {proposed_ply.name}")
        except Exception as e:
            print(f"  ❌ 导出本文方法点云失败: {e}")
        
        # 对比点云（蓝=基线，红=本文）
        print(f"\n  生成对比点云...")
        baseline_marked = baseline_points.copy()
        proposed_marked = proposed_points.copy()
        
        baseline_colors_marked = np.ones((len(baseline_points), 3), dtype=np.uint8)
        baseline_colors_marked[:] = [100, 150, 255]  # 蓝色
        
        proposed_colors_marked = np.ones((len(proposed_points), 3), dtype=np.uint8)
        proposed_colors_marked[:] = [255, 100, 100]  # 红色
        
        combined_points = np.vstack([baseline_marked, proposed_marked])
        combined_colors = np.vstack([baseline_colors_marked, proposed_colors_marked])
        
        combined_ply = output_dir / "reconstruction_comparison.ply"
        try:
            trimesh.PointCloud(combined_points, colors=combined_colors).export(str(combined_ply))
            print(f"  ✓ {combined_ply.name}")
            print(f"    - 蓝色点: 基线方法 ({len(baseline_points)} 点)")
            print(f"    - 红色点: 本文方法 ({len(proposed_points)} 点)")
        except Exception as e:
            print(f"  ❌ 导出对比点云失败: {e}")
    
    return results


def export_separate_trajectories(baseline_colmap: Path,
                                proposed_colmap: Path,
                                output_dir: Path):
    """
    分别导出两种方法的摄像机轨迹（用于轨迹对比）
    """
    if not COLMAP_AVAILABLE:
        return
    
    print(f"\n📹 导出摄像机轨迹对比...")
    
    baseline_recon = load_colmap_reconstruction(baseline_colmap)
    proposed_recon = load_colmap_reconstruction(proposed_colmap)
    
    if not baseline_recon or not proposed_recon:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取基线的摄像机位置
    baseline_centers = []
    for image_id in sorted(baseline_recon.images.keys()):
        image = baseline_recon.images[image_id]
        c = -image.cam_from_world.rotation.matrix().T @ image.cam_from_world.translation
        baseline_centers.append(c)
    
    # 提取本文方法的摄像机位置
    proposed_centers = []
    for image_id in sorted(proposed_recon.images.keys()):
        image = proposed_recon.images[image_id]
        c = -image.cam_from_world.rotation.matrix().T @ image.cam_from_world.translation
        proposed_centers.append(c)
    
    if baseline_centers and proposed_centers:
        baseline_traj = np.array(baseline_centers)
        proposed_traj = np.array(proposed_centers)
        
        # 导出为PLY（轨迹作为线段）
        if TRIMESH_AVAILABLE:
            # 创建轨迹点云
            baseline_traj_ply = output_dir / "trajectory_baseline.ply"
            proposed_traj_ply = output_dir / "trajectory_proposed.ply"
            
            baseline_colors_traj = np.ones((len(baseline_traj), 3), dtype=np.uint8)
            baseline_colors_traj[:] = [100, 150, 255]
            
            proposed_colors_traj = np.ones((len(proposed_traj), 3), dtype=np.uint8)
            proposed_colors_traj[:] = [255, 100, 100]
            
            try:
                trimesh.PointCloud(baseline_traj, colors=baseline_colors_traj).export(str(baseline_traj_ply))
                print(f"  ✓ {baseline_traj_ply.name}")
            except:
                pass
            
            try:
                trimesh.PointCloud(proposed_traj, colors=proposed_colors_traj).export(str(proposed_traj_ply))
                print(f"  ✓ {proposed_traj_ply.name}")
            except:
                pass
        
        # 保存轨迹数据（用于后续分析）
        traj_data = {
            "baseline_trajectory": baseline_traj.tolist(),
            "proposed_trajectory": proposed_traj.tolist(),
            "num_frames": len(baseline_traj)
        }
        
        traj_json = output_dir / "trajectories.json"
        with open(traj_json, "w") as f:
            json.dump(traj_data, f, indent=2)
        print(f"  ✓ {traj_json.name}")


def main():
    parser = argparse.ArgumentParser(
        description="对比Token分割方法的三维重建质量（点云对比）"
    )
    parser.add_argument(
        "--baseline_colmap",
        type=str,
        required=True,
        help="基线方法的COLMAP稀疏重建目录（sparse/目录的路径）"
    )
    parser.add_argument(
        "--proposed_colmap",
        type=str,
        required=True,
        help="本文方法的COLMAP稀疏重建目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/tests_result/reconstruction_comparison",
        help="输出目录（用于保存点云和对比结果）"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    baseline_path = Path(args.baseline_colmap)
    proposed_path = Path(args.proposed_colmap)
    
    # 验证路径
    if not baseline_path.exists():
        print(f"❌ 基线COLMAP目录不存在: {baseline_path}")
        return
    
    if not proposed_path.exists():
        print(f"❌ 本文方法COLMAP目录不存在: {proposed_path}")
        return
    
    print("\n" + "="*80)
    print("Token分割方法重建质量对比分析")
    print("="*80)
    
    # 执行对比
    results = compare_reconstructions(baseline_path, proposed_path, output_dir)
    
    if results:
        # 导出轨迹
        export_separate_trajectories(baseline_path, proposed_path, output_dir)

        # 保存结果JSON
        results_json = output_dir / "comparison_results.json"
        with open(results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ 结果保存至: {results_json}")
    
    print("\n" + "="*80)
    print("✅ 重建对比完成")
    print("="*80)
    print(f"\n📁 输出目录: {output_dir}")
    print("\n生成的文件可用以下工具查看:")
    print("  • CloudCompare: 查看PLY点云对比")
    print("  • Meshlab: 点云编辑和分析")
    print("  • COLMAP GUI: 查看完整的SfM重建结果")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
