# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Automatic Mask Placement - Single and Batch Generation Script

Usage:

# Single generation
python3 -m scripts.anomaly_gen.automatic_mask_placement \
        --submask path/to/submask.png \
        --n 1 \
        --roi_image path/to/roi.png \
        --output_base_dir path/to/output \
        --seed 42

# Batch generation with seed range
python3 -m scripts.anomaly_gen.automatic_mask_placement \
        --submask path/to/submask.png \
        --n 1 \
        --roi_image path/to/roi.png \
        --output_base_dir path/to/output \
        --seed_range 1 10 \
        --parallel_workers 4

# Batch generation with seed list
python3 -m scripts.anomaly_gen.automatic_mask_placement \
        --submask path/to/submask.png \
        --n 1 \
        --roi_image path/to/roi.png \
        --output_base_dir path/to/output \
        --seed_list "1,5,10,42"

# Disable ROI separation (treat entire image as one ROI)
python3 -m scripts.anomaly_gen.automatic_mask_placement \
        --submask path/to/submask.png \
        --n 1 \
        --roi_image path/to/roi_with_dots.png \
        --output_base_dir path/to/output \
        --seed 42 \
        --no_separate_rois

# Set max retry attempts per mask
python3 -m scripts.anomaly_gen.automatic_mask_placement \
        --submask path/to/submask.png \
        --n 1 \
        --roi_image path/to/roi.png \
        --output_base_dir path/to/output \
        --seed 42 \
        --max_retry_per_mask 20

# Override image dimensions (use specified size instead of submask size)
python3 -m scripts.anomaly_gen.automatic_mask_placement \
        --submask path/to/submask.png \
        --n 1 \
        --roi_image path/to/roi.png \
        --output_base_dir path/to/output \
        --seed 42 \
        --image_size 1920 1080

python3 -m scripts.anomaly_gen.automatic_mask_placement -h
"""

import argparse
import os
import sys
import time
import random
from PIL import Image
from typing import List, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from imaginaire.utils import log

from automatic_mask_placement import (
    AutomaticMaskPlacement,
    AugmentationParams,
    AlignmentPoint
)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Automatic Mask Placement Program")
    
    # Required arguments
    parser.add_argument("--submask", required=True, help="Path to submask image")
    parser.add_argument("--output_base_dir", required=True, help="Base output directory")
    
    # N parameter
    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--n", type=int, help="Number of instances to generate (fixed for all seeds)")
    n_group.add_argument("--n_range", nargs=2, type=int, metavar=('MIN', 'MAX'), help="Random N range [min, max] per seed")
    
    # ROI arguments
    parser.add_argument("--rois", help="JSON file containing ROI definitions")
    parser.add_argument("--roi_image", nargs="*", help="Binary ROI image file(s) for legal regions")
    parser.add_argument("--illegal_roi_image", nargs="*", help="Binary ROI image file(s) for illegal regions")
    
    # Seed control (mutually exclusive)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=str, help="Single seed (integer or 'None' for random)")
    seed_group.add_argument("--seed_range", nargs=2, type=int, metavar=('START', 'END'), help="Seed range [start, end] (inclusive)")
    seed_group.add_argument("--seed_list", type=str, help="Comma-separated seed list (e.g., '1,5,10,42')")
    
    # Batch options
    parser.add_argument("--parallel_workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    
    # Optional arguments
    parser.add_argument("--save_visualize", action="store_true", help="Save ROI visualization")
    parser.add_argument("--save_roi_binaries", action="store_true", help="Save binary images from ROI definitions")
    parser.add_argument("--save_separated_rois", action="store_true", help="Save separated ROI masks")
    parser.add_argument("--aug_config", help="Path to augmentation config file")
    parser.add_argument("--save_aug_config", action="store_true", help="Save augmentation parameters")
    parser.add_argument("--roi_alignment_point", default="random", 
                       choices=["center", "top_left", "top_right", "bottom_left", "bottom_right", 
                               "top_center", "bottom_center", "left_center", "right_center", "random"],
                       help="Alignment point for mask placement within ROI")
    parser.add_argument("--submask_alignment_point", default=None,
                       choices=["center", "top_left", "top_right", "bottom_left", "bottom_right", 
                               "top_center", "bottom_center", "left_center", "right_center", "random"],
                       help="Fixed point mode for submask augmentation")
    parser.add_argument("--strict_alignment", action="store_true", 
                       help="Enable strict alignment mode")
    parser.add_argument("--save_cropped_submask", action="store_true", help="Save cropped submask")
    parser.add_argument("--save_augmented_masks", action="store_true", help="Save augmented masks")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum area threshold for ROI filtering (default: 10)")
    parser.add_argument("--max_retry_per_mask", type=int, default=10, 
                       help="Maximum retry attempts if mask disappears after ROI clipping (default: 10)")
    parser.add_argument("--no_separate_rois", action="store_true", 
                       help="Disable automatic separation of connected regions in ROI images. "
                            "When enabled, the entire ROI image will be treated as a single ROI instead of "
                            "separating it into multiple individual ROIs (default: separation is enabled)")
    
    # Image size override
    parser.add_argument("--image_size", nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help="Override image size as WIDTH HEIGHT (default: use submask dimensions)")
    
    return parser.parse_args()


def get_seed_list(args) -> List[Optional[int]]:
    """Get list of seeds to process"""
    if args.seed is not None:
        # Single seed mode
        if args.seed.lower() == 'none':
            return [None]
        else:
            try:
                return [int(args.seed)]
            except ValueError:
                raise ValueError(f"Invalid seed value: '{args.seed}'. Use an integer or 'None'")
    
    elif args.seed_range:
        seed_start, seed_end = args.seed_range
        if seed_start > seed_end:
            raise ValueError(f"Seed range start ({seed_start}) must be <= end ({seed_end})")
        return list(range(seed_start, seed_end + 1))
    
    elif args.seed_list:
        try:
            seeds = [int(s.strip()) for s in args.seed_list.split(',')]
            return seeds
        except ValueError as e:
            raise ValueError(f"Invalid seed list format: {e}")
    
    else:
        raise ValueError("Either --seed, --seed_range, or --seed_list must be provided")


def get_n_mapping(args, seeds: List[Optional[int]]) -> Dict[Optional[int], int]:
    """Get N value for each seed"""
    if args.n is not None:
        return {seed: args.n for seed in seeds}
    
    elif args.n_range:
        min_n, max_n = args.n_range
        if min_n <= 0 or max_n <= 0:
            raise ValueError(f"N range values must be positive, got [{min_n}, {max_n}]")
        if min_n > max_n:
            raise ValueError(f"N range min must be <= max, got [{min_n}, {max_n}]")
        
        mapping = {}
        for seed in seeds:
            if seed is None:
                # For None seed, use a random N
                mapping[seed] = random.randint(min_n, max_n)
            else:
                # Deterministic random based on seed
                seed_random = random.Random(seed)
                mapping[seed] = seed_random.randint(min_n, max_n)
        
        return mapping
    
    else:
        raise ValueError("Either --n or --n_range must be provided")


def process_single_seed(args, seed: Optional[int], n_value: int, output_dir: str, show_progress: bool = True) -> bool:
    """Process a single seed"""
    try:
        # Get image size from args or auto-detect from submask
        submask = Image.open(args.submask)
        
        # Use user-specified dimensions if provided, otherwise use submask dimensions
        if args.image_size is not None:
            image_width, image_height = args.image_size
            if show_progress:
                log.info(f"Using specified image size: {image_width}x{image_height}")
        else:
            image_width = submask.width
            image_height = submask.height
            if show_progress:
                log.info(f"Auto-detected image size from submask: {image_width}x{image_height}")
        
        # Note: Submask resizing is now handled automatically in core.py
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load augmentation parameters
        augmentation_params = None
        if args.aug_config:
            augmentation_params = AugmentationParams.from_config_file(args.aug_config)
            if show_progress:
                log.info(f"Loaded augmentation parameters from: {args.aug_config}")
                log.info(f"Config settings: {augmentation_params.report_dynamic_operations()}")
        elif show_progress:
            log.info("Using default augmentation parameters")
        
        # Parse alignment points
        roi_alignment_point = AlignmentPoint(args.roi_alignment_point)
        submask_alignment_point = AlignmentPoint(args.submask_alignment_point) if args.submask_alignment_point else None
        
        # Create automatic mask placement instance
        amp = AutomaticMaskPlacement(
            image_width, image_height, 
            augmentation_params, 
            roi_alignment_point, 
            submask_alignment_point, 
            args.strict_alignment, 
            seed,
            max_retry_per_mask=args.max_retry_per_mask,
            separate_rois=not args.no_separate_rois  # Convert --no_separate_rois flag to separate_rois parameter
        )
        
        # Load ROI data
        if show_progress and args.roi_image:
            roi_sep_status = "disabled" if args.no_separate_rois else "enabled"
            log.info(f"ROI separation: {roi_sep_status}")
        
        amp.load_combined_rois(
            json_path=args.rois,
            roi_image_paths=args.roi_image,
            illegal_image_paths=args.illegal_roi_image,
            min_area=args.min_area
        )
        
        # Save optional outputs
        if args.save_roi_binaries:
            if show_progress:
                log.info("\nSaving ROI binary images...")
            roi_binary_path = os.path.join(output_dir, "roi_binaries")
            amp.create_roi_binary_images(roi_binary_path)
        
        if args.save_separated_rois:
            if show_progress:
                log.info("\nSaving separated ROI masks...")
            separated_roi_path = os.path.join(output_dir, "separated_roi_masks")
            amp.save_separated_roi_masks(separated_roi_path)
        
        if args.save_visualize:
            amp.visualize_rois(os.path.join(output_dir, "roi_visualization.png"))
        
        if args.save_aug_config:
            config_path = os.path.join(output_dir, "augmentation_config.json")
            amp.mask_augmentor.params.save_to_config_file(config_path)
            if show_progress:
                log.info(f"Saved augmentation parameters to: {config_path}")
        
        # Process submask (resize is handled automatically in core.py)
        output_paths = amp.process_submask(
            args.submask, n_value, output_dir, 
            args.save_cropped_submask, 
            args.save_augmented_masks, 
            args.strict_alignment
        )
        
        if show_progress:
            log.info(f"\nGenerated {len(output_paths)} augmented mask instances in '{output_dir}'")
        
        return True
        
    except Exception as e:
        if show_progress:
            log.error(f"Error: {e}")
        raise


def run_single_seed_wrapper(args_tuple):
    """Wrapper for parallel processing"""
    args, seed, n_value, output_dir = args_tuple
    start_time = time.time()
    try:
        process_single_seed(args, seed, n_value, output_dir, show_progress=False)
        elapsed = time.time() - start_time
        return seed, True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start_time
        return seed, False, elapsed, str(e)


def main():
    """Main function"""
    args = get_args()
    
    # Validate input arguments
    if not args.rois and not args.roi_image:
        raise ValueError("Either --rois or --roi_image must be provided")
    
    # Get seeds and N mapping
    seeds = get_seed_list(args)
    n_mapping = get_n_mapping(args, seeds)
    
    # Detect single seed mode
    is_single_seed = len(seeds) == 1
    
    # Display configuration
    if is_single_seed:
        seed = seeds[0]
        n_value = n_mapping[seed]
        seed_str = "None" if seed is None else str(seed)
        
        # Create seed subdirectory (consistent with batch mode)
        if seed is None:
            output_dir = os.path.join(args.output_base_dir, "seed_None")
        else:
            output_dir = os.path.join(args.output_base_dir, f"seed_{seed:04d}")
        
        log.info(f"Running automatic mask placement (single mode)")
        log.info(f"Seed: {seed_str}")
        log.info(f"Instances: {n_value}")
        log.info(f"Output directory: {output_dir}")
        log.info("")
        
        # Single seed: run directly with output
        start_time = time.time()
        success = process_single_seed(args, seed, n_value, output_dir, show_progress=True)
        elapsed = time.time() - start_time
        
        if success:
            log.info(f"\nCompleted in {elapsed:.1f}s (seed: {seed_str})")
            log.info(f"Output: {output_dir}")
        else:
            log.error(f"\nGeneration failed")
            sys.exit(1)
    
    else:
        # Batch mode
        log.info(f"Starting batch automatic mask placement generation")
        log.info(f"Seeds to process: {len(seeds)}")
        
        if args.n is not None:
            log.info(f"Instances per seed: {args.n} (same for all)")
        else:
            n_values = list(n_mapping.values())
            log.info(f"Instances per seed: {min(n_values)}-{max(n_values)} (variable)")
        
        log.info(f"Output directory: {args.output_base_dir}")
        log.info(f"Parallel workers: {args.parallel_workers}")
        log.info("")
        
        # Create base output directory
        os.makedirs(args.output_base_dir, exist_ok=True)
        
        # Track results
        successful_seeds = []
        failed_seeds = []
        total_start_time = time.time()
        
        if args.parallel_workers == 1:
            # Sequential processing
            for i, seed in enumerate(seeds, 1):
                n_value = n_mapping[seed]
                seed_str = "None" if seed is None else str(seed)
                output_dir = os.path.join(args.output_base_dir, f"seed_{seed_str if seed is None else f'{seed:04d}'}")
                
                log.info(f"[{i}/{len(seeds)}] Processing seed {seed_str} (N={n_value})...")
                start_time = time.time()
                
                try:
                    process_single_seed(args, seed, n_value, output_dir, show_progress=False)
                    elapsed = time.time() - start_time
                    successful_seeds.append(seed)
                    log.info(f"Seed {seed_str} completed in {elapsed:.1f}s")
                except Exception as e:
                    elapsed = time.time() - start_time
                    failed_seeds.append(seed)
                    log.error(f"Seed {seed_str} failed after {elapsed:.1f}s")
                    log.error(f"   Error: {e}")
        
        else:
            # Parallel processing
            log.info(f"Running {args.parallel_workers} workers in parallel...")
            
            # Prepare tasks
            tasks = []
            for seed in seeds:
                n_value = n_mapping[seed]
                seed_str = "None" if seed is None else f"{seed:04d}"
                output_dir = os.path.join(args.output_base_dir, f"seed_{seed_str}")
                tasks.append((args, seed, n_value, output_dir))
            
            with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
                futures = {executor.submit(run_single_seed_wrapper, task): task[1] for task in tasks}
                
                for i, future in enumerate(as_completed(futures), 1):
                    seed, success, elapsed, error = future.result()
                    seed_str = "None" if seed is None else str(seed)
                    
                    if success:
                        successful_seeds.append(seed)
                        log.info(f"[{i}/{len(seeds)}] Seed {seed_str} (N={n_mapping[seed]}) completed in {elapsed:.1f}s")
                    else:
                        failed_seeds.append(seed)
                        log.error(f"[{i}/{len(seeds)}] Seed {seed_str} (N={n_mapping[seed]}) failed after {elapsed:.1f}s")
                        if error:
                            log.error(f"   Error: {error}")
        
        # Summary
        total_elapsed = time.time() - total_start_time
        total_images = sum(n_mapping[seed] for seed in successful_seeds)
        
        log.info("")
        log.info("=" * 60)
        log.info("BATCH GENERATION SUMMARY")
        log.info("=" * 60)
        log.info(f"Successful: {len(successful_seeds)}/{len(seeds)} seeds")
        log.info(f"Failed: {len(failed_seeds)}/{len(seeds)} seeds")
        log.info(f"Total time: {total_elapsed:.1f}s")
        log.info(f"Total images generated: {total_images}")
        log.info(f"Output directory: {args.output_base_dir}")
        
        if failed_seeds:
            log.warning(f"\nFailed seeds: {failed_seeds}")
        if successful_seeds:
            log.info(f"\nSuccessful seeds: {sorted(successful_seeds)}")


if __name__ == "__main__":
    main()
