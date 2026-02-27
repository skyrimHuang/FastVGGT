#!/bin/bash
# Monitor ablation study progress

echo "Monitoring Ablation Study Progress..."
echo "========================================"
echo ""

while true; do
    clear
    echo "FastVGGT Ablation Study - Progress Monitor"
    echo "=========================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check each experiment
    experiments=("baseline" "norm_only" "threshold_75" "threshold_85" "combined_50" "combined_75")
    
    for exp in "${experiments[@]}"; do
        exp_dir="tests/tests_result/ablation/$exp"
        
        if [ -d "$exp_dir/input_frame_100" ]; then
            # Count completed scenes
            scene_count=$(find "$exp_dir/input_frame_100" -maxdepth 1 -type d -name "scene*" | wc -l)
            
            if [ -f "$exp_dir/input_frame_100/average_metrics.json" ]; then
                status="✓ COMPLETED"
                cd=$(jq -r '.chamfer_distance' "$exp_dir/input_frame_100/average_metrics.json" 2>/dev/null || echo "N/A")
                ate=$(jq -r '.ate' "$exp_dir/input_frame_100/average_metrics.json" 2>/dev/null || echo "N/A")
                echo "$exp: $status (5/5 scenes) - CD: $cd, ATE: $ate"
            elif [ $scene_count -gt 0 ]; then
                status="⏳ IN PROGRESS"
                echo "$exp: $status ($scene_count/5 scenes)"
            else
                status="⏸️  NOT STARTED"
                echo "$exp: $status"
            fi
        elif [ -d "$exp_dir" ]; then
            if [ -f "$exp_dir/average_metrics.json" ]; then
                # Old format (no input_frame subdirectory)
                status="✓ COMPLETED (old format)"
                cd=$(jq -r '.chamfer_distance' "$exp_dir/average_metrics.json" 2>/dev/null || echo "N/A")
                echo "$exp: $status - CD: $cd"
            else
                status="⏸️  NOT STARTED"
                echo "$exp: $status"
            fi
        else
            status="⏸️  NOT STARTED"
            echo "$exp: $status"
        fi
    done
    
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 10
done
