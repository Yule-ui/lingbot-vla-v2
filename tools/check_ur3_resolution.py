"""CPU checks for real rectangular UR3 observations and auxiliary token spans."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from ur3_adapters import install_data_adapters
from lingbotvla.data.vla_data.base_dataset import VLADataset
from lingbotvla.data.vla_data.utils import resolve_image_size
from lingbotvla.models import build_processor
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import FlowMatchingV2
from lingbotvla.models.vla.vision_models.module_utils import get_depth_target


def main():
    manifest = install_data_adapters(ROOT / 'assets/training_data/ur3_20261016_split_seed42.json', 'train')
    cfg = yaml.safe_load((ROOT / 'configs/vla/ur3_20261016/data_check.yaml').read_text())
    data = dict(cfg['data'])
    for key in ['joints', 'norm_type']:
        data[key] = [str(value) for value in data[key]]
    model_config = LingbotVLAV2Config(**cfg['model'], **cfg['train'])
    processor = build_processor(cfg['model']['tokenizer_path'])
    assert resolve_image_size(SimpleNamespace(img_size=256)) == (256, 256)
    class DepthTeacher:
        def infer(self, images, **kwargs):
            return {'depth': torch.ones(images.shape[0], *images.shape[-2:])}
        def infer_feat(self, images, depth, **kwargs):
            return torch.ones(images.shape[0], 2, 14, 18), None
    # The aspect-preserving teacher emits 252 patches; the head needs 256.
    teacher = DepthTeacher()
    targets, _ = get_depth_target('MoRGBD', (teacher, teacher),
                                  torch.zeros(1, 1, 3, 480, 640, dtype=torch.uint8))
    assert targets.shape == (1, 256, 2)
    torch.testing.assert_close(targets, torch.ones_like(targets))
    results = []
    for height, width in [(256, 256), (288, 384), (480, 640)]:
        data_config = SimpleNamespace(**data, image_height=height, image_width=width)
        dataset = VLADataset(manifest['dataset_root'], data['data_name'], data_config,
            data['robot_config_root'], config=model_config, processor=processor,
            image_size=resolve_image_size(data_config), chunk_size=50,
            use_depth_align=True, use_future_image=True)
        item = dataset[0]
        count = (height // 32) * (width // 32)
        assert item['image_grid_thw'].tolist() == [[1, height // 16, width // 16]] * 3
        assert item['images'].shape == (3, count * 4, 1536)
        assert item['pil_images'].shape[-2:] == (height, width)
        assert item['future_pil_images'].shape[-2:] == (height, width)
        assert item['state'].shape == (55,) and item['actions'].shape == (50, 55)
        assert int(item['action_joint_mask'].sum()) == 8

        # Sentinel values distinguish the complete image block from query tokens.
        flow = FlowMatchingV2.__new__(FlowMatchingV2)
        torch.nn.Module.__init__(flow)
        flow.config = SimpleNamespace(qwen3vl_use_vision_boundaries=True)
        flow._image_patch_token_count = count
        flow.num_task_tokens = 8
        flow.use_future_depth = True
        flow.use_future_video = True
        flow.use_future_video_cls = False
        flow.use_future_video_patch = True
        flow.future_video_share_future_depth_query = True
        prefix_length = 3 * (count + 2) + 72 + 16
        hidden = torch.arange(prefix_length).reshape(1, -1, 1)
        image = flow._first_image_tokens(hidden)
        assert image.flatten().tolist() == list(range(1, count + 1))
        queries = flow._current_depth_task_tokens(hidden)
        assert queries.flatten().tolist() == list(range(prefix_length - 16, prefix_length - 8))
        future = flow._future_depth_task_tokens(hidden)
        assert future.flatten().tolist() == list(range(prefix_length - 8, prefix_length))
        results.append({'height': height, 'width': width, 'tokens_per_image': count,
                        'images_shape': list(item['images'].shape),
                        'img_masks': item['img_masks'].tolist(),
                        'auxiliary_image_and_query_spans_passed': True})
    report = ROOT / 'docs/ur3_resolution_checks.json'
    report.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
