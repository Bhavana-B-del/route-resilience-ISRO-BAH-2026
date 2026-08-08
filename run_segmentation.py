import os
import numpy as np
from PIL import Image
from train_pathmamba_v2 import main as run_training
from inference import load_pathmamba_and_tiled_infer
from model_io import PathMambaModel
from skeletonization import run_skeletonization
from healing import run_graph_extraction
from centrality import compute_centrality_measures
from scenarios import simulate_failure_scenarios
from flood_simulation import assess_flood_impact
from resilience_index import compute_resilience_index
import importlib.util

def setup_env():
    os.system("pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")
    os.system("pip install causal-conv1d==1.4.0 mamba-ssm==2.2.2 --no-deps --no-build-isolation")

    mamba_path = importlib.util.find_spec("mamba_ssm").origin
    mamba_dir = os.path.dirname(mamba_path)

    with open(os.path.join(mamba_dir, "__init__.py"), "r") as f:
        content = f.read()
    content = content.replace("from .mamba_model import MambaLMHeadModel", "")
    with open(os.path.join(mamba_dir, "__init__.py"), "w") as f:
        f.write(content)

def select_input_area():
    while True:
        print("Please provide the path to the input image:")
        input_path = input("> ")
        if os.path.exists(input_path):
            return input_path
        else:
            print(f"File not found: {input_path}. Please try again.")

def run_segmentation(input_path):
    # Run the training pipeline
    run_training()

    # Load the trained model and run tiled inference
    model_path = os.path.join(os.environ["OUTPUT_ROOT"], "stage_b_snapshot.pth")
    image_gray_uint8 = np.array(Image.open(input_path).convert("L"))  # Load grayscale image
    segmentation = load_pathmamba_and_tiled_infer(
        checkpoint_path=model_path,
        image_gray_uint8=image_gray_uint8,
        pathmamba_cls=PathMambaModel,
        device="cuda",
        tile_size=512,
        exclusion_band_px=128,
        batch_size=4,
        transform=None,
        crs=None,
        source_gsd_m=0.28,
        glcm_norm_value=None,
        use_fast_glcm_proxy=False,
    )
    return segmentation

if __name__ == '__main__':
    setup_env()
    input_area = select_input_area()

    print("Running segmentation...")
    segmentation = run_segmentation(input_area)
    print("Segmentation completed.")

    print("Running skeletonization...")
    skeleton = run_skeletonization(segmentation)
    print("Skeletonization completed.")

    print("Extracting graph...")
    graph = run_graph_extraction(skeleton)
    print("Graph extraction completed.")

    print("Computing centrality measures...")
    centralities = compute_centrality_measures(graph)
    print("Centrality measures computed.")

    print("Simulating failure scenarios...")
    occluded_graphs = simulate_failure_scenarios(graph, centralities)
    print("Failure scenarios simulated.")

    print("Assessing flood impact...")
    flood_impacts = [assess_flood_impact(g) for g in occluded_graphs]
    print("Flood impact assessed.")

    print("Computing resilience index...")
    resilience_scores = compute_resilience_index(graph, occluded_graphs, flood_impacts)
    print(f"Resilience Index: {resilience_scores}")