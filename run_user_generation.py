import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image
import torch

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


def parse_args():
    parser = argparse.ArgumentParser(description="Run TRELLIS.2 image-to-3D generation and export GLB.")
    parser.add_argument("image", type=Path, help="Input image. RGBA alpha is used as the foreground mask.")
    parser.add_argument("--output", type=Path, default=None, help="Output GLB path.")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B", help="Model repo or local model path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline-type", default=None, choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--no-remesh", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image
    output_path = args.output
    if output_path is None:
        output_path = Path("users") / "outputs" / f"{image_path.stem}.glb"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    print(f"Image mode={image.mode} size={image.size} alpha={'A' in image.getbands()}")

    print(f"Loading pipeline: {args.model}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    print("Running image-to-3D generation")
    mesh = pipeline.run(
        image,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        max_num_tokens=args.max_num_tokens,
    )[0]

    # Keep mesh under nvdiffrast's face-count limit before texture baking.
    mesh.simplify(16_777_216, verbose=True)

    print(f"Exporting GLB: {output_path}")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=not args.no_remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
        use_tqdm=True,
    )
    glb.export(output_path, extension_webp=True)
    print(f"Wrote {output_path}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
