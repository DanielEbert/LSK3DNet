import os
import argparse
import numpy as np
import torch
from easydict import EasyDict
import json
import time
import pypcd4

from utils.load_util import load_yaml
from network.largekernel_model import get_model_class
from utils.load_save_util import load_checkpoint_old, load_checkpoint_model_mask
from utils.normalmap import compute_normals_range

DEFAULT_CONFIG_PATH = './config/lk-semantickitti_erk_finetune.yaml'

def run_inference():
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('--config_path', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--model_load_path', type=str, default=None,
                        help='Path to the trained model checkpoint (overrides config).')
    parser.add_argument('--points_path', type=str, default=None,
                        help='Optional path to a JSON file containing a list of [x,y,z,i] points, or .pcd file.')
    parser.add_argument('--num_points', type=int, default=50000,
                        help='Number of points for the dummy input sample (if --points_path is not used).')
    # Add projection parameters for normal calculation
    parser.add_argument('--fov_up', type=float, default=3.0, help='Sensor FOV up in degrees for normal calculation.')
    parser.add_argument('--fov_down', type=float, default=-25.0, help='Sensor FOV down in degrees for normal calculation.')
    parser.add_argument('--proj_h', type=int, default=64, help='Range image height for normal calculation.')
    parser.add_argument('--proj_w', type=int, default=900, help='Range image width for normal calculation.')

    args = parser.parse_args()

    print(f"Loading configuration from: {args.config_path}")
    if not os.path.exists(args.config_path):
        print(f"ERROR: Configuration file not found at {args.config_path}")
        return
    configs = EasyDict(load_yaml(args.config_path))

    final_model_load_path = args.model_load_path
    if final_model_load_path:
        print(f"Using model_load_path from command line: {final_model_load_path}")
    elif configs.model_params.get('model_load_path'):
        final_model_load_path = configs.model_params.model_load_path
        print(f"Using model_load_path from config file: {final_model_load_path}")
    else:
        print("ERROR: model_load_path not provided via command line or in the config file.")
        return

    if not os.path.exists(final_model_load_path):
        print(f"ERROR: Model checkpoint not found at {final_model_load_path}")
        return
    configs.model_params.model_load_path = final_model_load_path

    dataset_config = configs['dataset_params']
    model_config = configs['model_params']

    pytorch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {pytorch_device}")

    print(f"Instantiating model: {model_config['model_architecture']}")
    my_model = get_model_class(model_config['model_architecture'])(configs)

    print(f"Loading checkpoint from: {model_config['model_load_path']}")
    try:
        my_model = load_checkpoint_old(model_config['model_load_path'], my_model)
        print("Checkpoint loaded successfully (using load_checkpoint_old).")
    except Exception as e1:
        print(f"Failed to load with load_checkpoint_old: {e1}. Trying load_checkpoint_model_mask.")
        try:
            my_model, _ = load_checkpoint_model_mask(model_config['model_load_path'], my_model, pytorch_device)
            print("Checkpoint loaded successfully (using load_checkpoint_model_mask, mask ignored).")
        except Exception as e2:
            print(f"ERROR: Could not load checkpoint: {e2}")
            return

    my_model.to(pytorch_device)
    my_model.eval()

    if args.points_path:
        if args.points_path.endswith('.json'):
            with open(args.points_path, 'r') as f:
                points_list = json.load(f)
            raw_points_xyz_intensity_np = np.array(points_list, dtype=np.float32)
        elif args.points_path.endswith('.pcd'):
            raw_points_xyz_intensity_np = pypcd4.PointCloud.from_path(args.points_path).numpy()[:, :4]
        else:
            raise Exception(f'Unknown points_path file ending {args.points_path}, expected .json or .pcd')
        if raw_points_xyz_intensity_np.ndim != 2 or raw_points_xyz_intensity_np.shape[1] != 4:
            raise ValueError("JSON points should be a list of [x,y,z,i] lists.")
        num_actual_points = raw_points_xyz_intensity_np.shape[0]
        print(f"Loaded {num_actual_points} points from JSON.")
    else:
        print(f"Generating {args.num_points} dummy random points.")
        num_actual_points = args.num_points
        # Points: (N, 4) where (x,y,z,intensity)
        raw_points_xyz_intensity_np = np.random.rand(num_actual_points, 4).astype(np.float32)
        # Scale dummy points to be somewhat within typical Lidar ranges for better voxelization
        min_vol_np = np.array(dataset_config.min_volume_space, dtype=np.float32)
        max_vol_np = np.array(dataset_config.max_volume_space, dtype=np.float32)
        raw_points_xyz_intensity_np[:, 0] = raw_points_xyz_intensity_np[:, 0] * (max_vol_np[0] - min_vol_np[0]) + min_vol_np[0]
        raw_points_xyz_intensity_np[:, 1] = raw_points_xyz_intensity_np[:, 1] * (max_vol_np[1] - min_vol_np[1]) + min_vol_np[1]
        raw_points_xyz_intensity_np[:, 2] = raw_points_xyz_intensity_np[:, 2] * (max_vol_np[2] - min_vol_np[2]) + min_vol_np[2]
        raw_points_xyz_intensity_np[:, 3] = np.random.rand(num_actual_points).astype(np.float32) # Intensity
    
    if raw_points_xyz_intensity_np.shape[0] == 0:
        print("ERROR: No points to process.")
        return

    # compute_normals_range expects (N,4) numpy array where last col is intensity
    try:
        start_normal_time = time.time()
        normals_np = compute_normals_range(
            raw_points_xyz_intensity_np,
            fov_up=args.fov_up,
            fov_down=args.fov_down,
            proj_H=args.proj_h,
            proj_W=args.proj_w
        ) # Output is (N,3)
        end_normal_time = time.time()
        print(f"Normals calculated in {end_normal_time - start_normal_time:.4f} seconds.")
        if normals_np.shape[0] != raw_points_xyz_intensity_np.shape[0]:
            print(f"Warning: Number of normals ({normals_np.shape[0]}) does not match number of points ({raw_points_xyz_intensity_np.shape[0]}). This might indicate issues with range projection.")
            # Handle this mismatch, e.g. by taking only points for which normals were computed, or erroring.
            # For now, let's assume it works or we can proceed with a subset if necessary, though ideally they match.
            # This can happen if some points are outside the FoV for range projection.
            # The current compute_normals_range returns normals for original points based on projection indices.
    except Exception as e:
        print(f"Error during C++ normal calculation: {e}")
        print("Falling back to random normals for inference.")
        normals_np = np.random.rand(raw_points_xyz_intensity_np.shape[0], 3).astype(np.float32)
        normals_np = normals_np / np.linalg.norm(normals_np, axis=1, keepdims=True)

    points_tensor = torch.from_numpy(raw_points_xyz_intensity_np).to(pytorch_device)
    normals_tensor = torch.from_numpy(normals_np).to(pytorch_device)
    batch_indices = torch.zeros(points_tensor.shape[0], device=pytorch_device, dtype=torch.long)

    inference_data_dict = {
        'points': points_tensor,         # Shape: [N, 4] (x,y,z,intensity)
        'normal': normals_tensor,        # Shape: [N, 3]
        'batch_idx': batch_indices,      # Shape: [N]
        'batch_size': 1                  # Model's spconv tensor needs this
    }
    print(f"Prepared input: points shape {inference_data_dict['points'].shape}, normal shape {inference_data_dict['normal'].shape}")

    print("Running inference...")
    with torch.no_grad():
        start_time = time.time()
        output_dict = my_model(inference_data_dict)
        end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.4f} seconds.")

    logits = output_dict['logits'] 
    predicted_labels_for_output_points = torch.argmax(logits, dim=1)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Predicted labels shape for output points: {predicted_labels_for_output_points.shape}")

    num_to_show = min(10, predicted_labels_for_output_points.shape[0])
    if num_to_show > 0:
        print(f"\nFirst {num_to_show} predicted labels for output points:")
        print(predicted_labels_for_output_points.cpu().numpy()[:num_to_show])
    else:
        print("\nNo output points generated from dummy input/JSON (check voxelization or input data scaling).")

if __name__ == '__main__':
    run_inference()

