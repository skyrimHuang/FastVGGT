import sys
import torch
sys.path.insert(0, '.')
sys.path.insert(0, './eval')
from vggt.models.vggt import VGGT
from vggt.utils.keyframe_filter import KeyframeFilter
from data import SevenScenes

# Load model
model = VGGT(merging=25, merge_ratio=0.0, enable_point=True, enable_depth=True, enable_camera=True)
model.load_state_dict(torch.load('/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt', map_location='cpu'), strict=False)
model = model.to('cuda:0').eval()

# Load data
dataset = SevenScenes(split='test', ROOT='/home/hba/Documents/Dataset/7_scenes', resolution=(518,392), num_seq=1, full_video=True, kf_every=1)
views = dataset[0]
imgs = torch.stack([v['img'] for v in views[:10]])
imgs = (imgs + 1.0)/2.0
imgs = imgs.unsqueeze(0).to('cuda:0', dtype=torch.float32)

# Test filter
filter_model = KeyframeFilter(aggregator=model.aggregator, threshold=0.3)

print('Testing KeyframeFilter...')
print(f'Input shape: {imgs.shape}')

try:
    result = filter_model(imgs)
    print('✓ Success!')
    print(f'  filtered_images: {result["filtered_images"].shape}')
    print(f'  patch_tokens: {result["patch_tokens"].shape}')
    print(f'  keyframe_indices: {result["keyframe_indices"]}')
    print(f'  compression_ratio: {result["stats"]["compression_ratio"]}')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
