#!/bin/bash
# Monitor testing progress for both baseline and improved methods

echo "======================================================"
echo "Monitoring Test Progress (Ctrl+C to exit)"
echo "======================================================"

while true; do
    clear
    echo "======================================================"
    echo "Test Progress Monitor - $(date '+%H:%M:%S')"
    echo "======================================================"
    
    # Check baseline
    echo -e "\n[BASELINE - Grid-Based Split]"
    baseline_dir="tests/tests_result/baseline_grid_5scenes/input_frame_100"
    if [ -d "$baseline_dir" ]; then
        scene_count=$(find "$baseline_dir" -mindepth 1 -maxdepth 1 -type d -name "scene*" | wc -l)
        echo "  ✓ Completed scenes: $scene_count / 5"
        
        if [ -f "$baseline_dir/average_metrics.json" ]; then
            echo "  ✓ Average metrics calculated"
            cd_val=$(jq -r '.chamfer_distance' "$baseline_dir/average_metrics.json" 2>/dev/null || echo "N/A")
            ate_val=$(jq -r '.ate' "$baseline_dir/average_metrics.json" 2>/dev/null || echo "N/A")
            time_val=$(jq -r '.inference_time_ms' "$baseline_dir/average_metrics.json" 2>/dev/null || echo "N/A")
            echo "    CD: $cd_val | ATE: $ate_val | Time: ${time_val}ms"
        fi
    else
        echo "  ⏳ Waiting for results..."
    fi
    
    # Check improved
    echo -e "\n[IMPROVED - Norm-Guided + Threshold]"
    improved_dir="tests/tests_result/improved_norm_guided_5scenes/input_frame_100"
    if [ -d "$improved_dir" ]; then
        scene_count=$(find "$improved_dir" -mindepth 1 -maxdepth 1 -type d -name "scene*" | wc -l)
        echo "  ✓ Completed scenes: $scene_count / 5"
        
        if [ -f "$improved_dir/average_metrics.json" ]; then
            echo "  ✓ Average metrics calculated"
            cd_val=$(jq -r '.chamfer_distance' "$improved_dir/average_metrics.json" 2>/dev/null || echo "N/A")
            ate_val=$(jq -r '.ate' "$improved_dir/average_metrics.json" 2>/dev/null || echo "N/A")
            time_val=$(jq -r '.inference_time_ms' "$improved_dir/average_metrics.json" 2>/dev/null || echo "N/A")
            echo "    CD: $cd_val | ATE: $ate_val | Time: ${time_val}ms"
        fi
    else
        echo "  ⏳ Waiting for results..."
    fi
    
    echo -e "\n======================================================"
    echo "Refreshing in 10 seconds... (Ctrl+C to stop)"
    echo "======================================================"
    
    sleep 10
done
