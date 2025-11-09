import wandb
from pathlib import Path
import json
from src.data.physionet_challange_datamodule import PhysionetDataModule

# import os
# os.environ["WANDB_MODE"]="offline"

import pandas as pd
import shutil
import os
import torch
import functools
import pytorch_lightning as pl
import logging
import pickle

from typing import NamedTuple
import tempfile
from captum.attr import visualization as viz
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb
from captum.attr import Occlusion, Saliency, GradientShap, IntegratedGradients, NoiseTunnel, Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from skimage import graph, segmentation, color, filters, data
from lime import lime_image

import torch
from PIL import Image
import torchvision.transforms as transforms
from time import time
from skimage import graph, segmentation, color, filters

from EnsembleXAI.Ensemble import normEnsembleXAI
from EnsembleXAI.Normalization import second_moment_normalize
from handlers import GiraffeHandler

from giraffe.inferece_tree import InferenceTree
from giraffe.tree import Tree

from src.data_model.fold_config import FoldConfig
from src.lit_models.guangzhou_models import GuangzhouLitModel
from src.data.from_df_image_dataset import FromDfImageDataset
from src.utils.format import get_formatted_datetime

import numpy as np
from skimage import graph, segmentation, color, filters, data
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import time
import random

import numpy as np
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set the seed for reproducibility in random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

model_path_base = "/net/pr2/projects/plgrid/plggpolsl5/xai_lit/explanations_new"


def save_numpy_dict(dictionary, filename):
    """
    Save a dictionary of NumPy arrays to a file, creating the directory if it does not exist.

    Parameters:
    dictionary (dict): The dictionary containing NumPy arrays to save.
    filename (str): The file path where the dictionary will be saved.

    Returns:
    None
    """
    if not isinstance(dictionary, dict):
        raise TypeError("The input must be a dictionary.")

    for key, value in dictionary.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"All values in the dictionary must be NumPy arrays. Key '{key}' does not have a NumPy array value."
            )

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    np.savez(filename, **dictionary)


def create_mask_from_image(image_array: np.ndarray, compactness: int = 30, n_segments: int = 400) -> np.ndarray:
    """
    Create a mask using RAG boundaries on an input image.

    Parameters:
    - image_array (np.ndarray): Input RGB image as a numpy array.
    - compactness (int): Compactness parameter for SLIC segmentation. Higher values make segments more square.
    - n_segments (int): The number of segments to generate.

    Returns:
    - np.ndarray: A numpy array representing the mask with regions as unique labels.
    """
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image_array)

    # Perform SLIC segmentation
    labels = segmentation.slic(image_array, compactness=compactness, n_segments=n_segments, start_label=1)

    # Compute edges using Sobel filter
    edges = filters.sobel(gray_image)

    # Build the Region Adjacency Graph (RAG)
    rag = graph.rag_boundary(labels, edges)

    # Create the mask based on regions (each region gets a unique integer label)
    mask = np.zeros_like(labels, dtype=np.int32)
    for region_id, nodes in enumerate(rag.nodes):
        for coord in np.argwhere(labels == nodes):
            mask[tuple(coord)] = region_id + 1  # Region IDs start from 1

    return mask


def compute_region_averages(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute the average RGB values of the pixels for each region defined in the mask.

    Parameters:
    - image_array (np.ndarray): Input RGB image as a numpy array.
    - mask (np.ndarray): Mask with unique integer labels for each region.

    Returns:
    - np.ndarray: Array of shape (N, 3) where N is the number of regions, and columns are average R, G, B values.
    """
    # Ensure the image and mask have compatible dimensions
    assert image_array.shape[:2] == mask.shape, "Image and mask dimensions do not match."

    # Get unique region labels
    unique_labels = np.unique(mask)

    # Prepare an array to hold the average RGB values for each region
    region_averages = np.zeros((len(unique_labels), 3))  # Shape: (num_regions, 3)

    for i, label in enumerate(unique_labels):
        # Get the mask for the current region
        region_mask = mask == label

        # Calculate the mean for each color channel (R, G, B)
        region_pixels = image_array[region_mask]
        region_averages[i] = region_pixels.max(axis=0)

    return region_averages


def visualize_region_averages(image_array: np.ndarray, mask: np.ndarray, region_averages: np.ndarray) -> np.ndarray:
    """
    Create a visualization of the regions with their average RGB values.

    Parameters:
    - image_array (np.ndarray): Original image as a numpy array.
    - mask (np.ndarray): Mask with unique integer labels for each region.
    - region_averages (np.ndarray): Average RGB values for each region.

    Returns:
    - np.ndarray: A new image where each region is colored with its average RGB value.
    """
    # Create a new image filled with average colors for each region
    recolored_image = np.zeros_like(image_array, dtype=np.uint8)
    unique_labels = np.unique(mask)

    for label, avg_color in zip(unique_labels, region_averages):
        recolored_image[mask == label] = avg_color.astype(np.uint8)

    return recolored_image


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help="Path to directory that will contain predictions",
        default="./outputs_inference",
    )
    parser.add_argument(
        "--explainer_config",
        default="/net/pr2/projects/plgrid/plggpolsl5/xai_lit/configs/explainers/basic_explainers.json",
    )
    parser.add_argument(
        "--max_elements",
        default=3,
    )
    parser.add_argument("--config_file")
    args = parser.parse_args()

    return args


def ensure_png_path(file_path):
    """
    Checks if the directory for a .png file exists, and creates it if it doesn't.

    Parameters:
        file_path (str): The full path to the .png file, including the file name.

    Returns:
        bool: True if the path exists or was successfully created, False if the path is invalid.
    """
    # Check if the file has a .png extension
    if not file_path.lower().endswith(".png"):
        print("Error: The file path must point to a .png file.")
        return False

    # Extract the directory path
    directory = os.path.dirname(file_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        try:
            # Create the directory structure
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False

    return True


# TODO: Przepisać Explainera, podzielić na headery
class Explainer:
    def __init__(self, lit_model, dataloader, explainers=[("saliency", {})], external_explainers={}, folder_name="explanations"):
        self.dataloader = dataloader
        self.lit_model = lit_model
        self.explainers_dict = explainers
        self.external_explainers = external_explainers
        self.folder_name = folder_name
        logging.error(self.folder_name)
        columns = (
            ["filename", "image", "pred", "ground_truth"]
            + list(self.explainers_dict.keys())
            + list(self.external_explainers.keys())
        )
        if len(explainers.keys()) > 1:
            columns.append("attrs_ensembled")
        logging.warning(f"{columns=}")
        self.pred_table = wandb.Table(columns=columns)

    def get_explainer(self, fn, explainer_type, kwargs):
        explainer = None
        logging.warning(f"{explainer=}")
        if "occlusion" in explainer_type:
            explainer = Occlusion(fn)
        elif "saliency" in explainer_type:
            explainer = Saliency(fn)
        elif "lime" in explainer_type:
            exp_eucl_distance = get_exp_kernel_similarity_function("euclidean", kernel_width=30)
            explainer = Lime(
                fn,
                interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
                similarity_func=exp_eucl_distance,
            )
        elif "gradientshap" in explainer_type:
            explainer = GradientShap(fn)
        elif "integrated_gradients" in explainer_type:
            explainer = IntegratedGradients(fn)
        explainer = NoiseTunnel(explainer)
        return explainer, kwargs, explainer_type

    def build_args(self, args, image):
        if "baselines" in args.keys() and isinstance(args["baselines"], bool):
            args["baselines"] = torch.zeros_like(image)
        if "strides" in args.keys() and isinstance(args["strides"], list):
            args["strides"] = tuple(args["strides"])
        if "sliding_window_shapes" in args.keys() and isinstance(args["sliding_window_shapes"], list):
            args["sliding_window_shapes"] = tuple(args["sliding_window_shapes"])

        return args

    def set_explainers(self, fn, image):
        self.explainers = {}
        for explainer_type, explainer_kwargs in self.explainers_dict.items():
            explainer, kwargs, explainer_type = self.get_explainer(fn, explainer_type, explainer_kwargs)
            built_kwargs = self.build_args(kwargs, image)
            self.explainers[explainer_type] = [explainer, built_kwargs]

    def _predict_fn(self, images):
        self.lit_model.eval()
        out = self.lit_model(images)
        out = torch.concat([1 - out, out], axis=-1)
        return out

    def explain_single_image(self, image, predicted_label, predict_fn=_predict_fn, target=0):
        """
        Generates an explanation for a single image.
        Uses the explainer model (Occlusion by default) to generate attribution.
        """
        self.set_explainers(predict_fn, image)
        attrs = {}

        device = torch.device("cuda:0")
        for explainer_name, explainer in self.explainers.items():
            logging.error(f"{explainer_name=}")
            free, total = torch.cuda.mem_get_info(device)
            time_start = time()
            attrs[explainer_name] = explainer[0].attribute(image, target=predicted_label, **explainer[1])
            logging.error(f"Attribution took: {time() - time_start}s, gpu memory -> free: {free}, total: {total}")

        external = {
            k: v(image, target, predicted_label, predict_fn)
            for k, v in self.external_explainers.items()
        }
        external_attrs  ={k:torch.tensor(v[0], device=device) for k,v in external.items()}
        external_figs  ={k:v[1] for k,v in external.items()}
        attrs.update(external_attrs)

        for key, attribution in attrs.items():
            if attribution.min() == attribution.max():
                attrs[key] += 1e4
        if len(attrs.keys()) > 1:
            concatenated_explanations = torch.stack(list(attrs.values()), dim=1).float()

            normalized_explanations = second_moment_normalize(concatenated_explanations)
            attrs_ensembled = normEnsembleXAI(normalized_explanations, aggregating_func="avg")
            attrs["attrs_ensembled"] = attrs_ensembled

        # Convert explanation to a format suitable for wandb.Image logging
        return {key: np.transpose(value.squeeze().cpu().detach().numpy(), (1, 2, 0)) for key, value in attrs.items()}, external_figs

    # move to CPU and reshape for logging

    def iterate_explain(self, predict_fn=_predict_fn):
        import logging

        for batch in tqdm(self.dataloader, desc="Explaining predictions", unit="batch"):
            image = batch[0].cuda()  # Move image to GPU
            original_image = np.transpose((image[0].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
            logging.error(f"Image std: {original_image.std()}, mean: {original_image.mean()}")
            target = batch[1].cuda()  # Move target to GPU if necessary
            # Get prediction
            pred = predict_fn(image).squeeze(0)
            predicted_label = int(torch.argmax(pred, dim=0).cpu().numpy())
            # predicted_label = 1

            attrs, figs = self.explain_single_image(image, predicted_label, predict_fn, target=target)
            t1 = viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image")[0]
            # t1.set_title("")
            plt.tight_layout()
            plots = {}
            for key, value in attrs.items():
                if False:
                    mask = create_mask_from_image(
                        image.detach().cpu().numpy().squeeze().transpose((1, 2, 0)), compactness=30, n_segments=100
                    )
                    logging.error(f"mask shape: {mask.shape}")
                    logging.error(f"attrs shape: {value.shape}")
                    region_averages = compute_region_averages(value, mask)
                    value = visualize_region_averages(value, mask, region_averages)
                    logging.error(f"attrs shape: {value.shape}")

                plot, ax = viz.visualize_image_attr(
                    value,
                    original_image,
                    "blended_heat_map",
                    "all",
                    show_colorbar=False,
                    outlier_perc=1,
                )
                ax.axis("off")
                plt.tight_layout()
                plots[key] = plot
            

            image_path = batch[2][0]
            row = [
                image_path,
                wandb.Image(image * 255),
                float(pred[1].cpu().detach().numpy()),
                float(target[0].cpu().detach().numpy()),
                *[wandb.Image(image) for image in plots.values()],
            ]
            logging.error(f"Length of the row: {len(row)}")
            print(row)
            self.pred_table.add_data(*row)
            for figname, fig in figs.items():
                plots[f"{figname}_original"] = fig
            global model_name


            self.pred_table.add_data(*row)
            global model_path_base, model_name

            base_path = f"/net/pr2/projects/plgrid/plggpolsl5/xai_lit/explanations2/{model_name}/{self.folder_name}"
            attrs["original_image"] = image.detach().cpu().numpy()
            attrs["path"] = np.array([image_path])
            sample_path = batch[2][0].replace("/", "_", -1).rstrip(".png")
            image_path_reduced = image_path.replace("/", "_", -1)

            save_numpy_dict(attrs, f"{base_path}/{sample_path}.npz")
            plots["original_image"] = t1
            for plot_name, plot in plots.items():
                image_path_reduced = image_path.replace("/", "_", -1)
                path_to_save = f"{base_path}/{image_path_reduced}/{plot_name}.png"
                ensure_png_path(path_to_save)
                logging.error(f"Saving to the path: {path_to_save}")
                plot.savefig(path_to_save, bbox_inches="tight", dpi=1000)
            plt.close()

    def iterate(self, predict_fn=_predict_fn):
        import logging

        for batch in tqdm(self.dataloader, desc="Explaining predictions", unit="batch"):
            image = batch[0].cuda()  # Move image to GPU
            original_image = np.transpose((image[0].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
            target = batch[1].cuda()  # Move target to GPU if necessary
            # Get prediction
            pred = predict_fn(image).squeeze(0)
            predicted_label = int(torch.argmax(pred, dim=0).cpu().numpy())

            row = [batch[2][0], float(pred[1].cpu().detach().numpy())]
            self.pred_table.add_data(*row)

    def upload(self, WANDB_RUN):
        WANDB_RUN.log({f"predictions_{WANDB_RUN.name}": self.pred_table})


def handle_wandb(wandb_config, wandb_additional_config):
    if wandb_config is not None:
        if "entity" in wandb_config and "project" in wandb_config:
            run = wandb.init(
                entity=wandb_config["entity"],
                project=wandb_config["project"],
                job_type=wandb_config.get("job_type", None),
                config=wandb_additional_config,
            )
            return run
        else:
            raise ValueError(f"Must provide both entity and project in wandb config, provided {wandb_config}")


def load_model(model_config, WANDB_RUN):
    if not isinstance(model_config, dict):  # assume link to wandb artifact or to giraffe architecture
        if WANDB_RUN is None:
            raise Exception("Weights and Biases config has to be provided to use model from registry")
        artifact = WANDB_RUN.use_artifact(model_config, type="model")
        artifact_dir = Path(artifact.download())
        model = GuangzhouLitModel.load_from_checkpoint(artifact_dir / "model.ckpt")
    else:
        architecture_path = model_config["architecture"]
        project_name = model_config["project"]
        pred_func_dict = GiraffeHandler.build_pred_func_dict(architecture_path, project_name)
        model = InferenceTree(tree_architecture_path=architecture_path, prediction_functions=pred_func_dict)

    return model


def prepare_folds_config(df):
    class FoldsConfig(NamedTuple):
        df: pd.DataFrame
        folds_train: list
        folds_val: list
        folds_test: list

    config = FoldsConfig(df=df, folds_test=["test"], folds_train=[], folds_val=[])

    return config


def get_lime_preds(image, target, label, predict_fn):
    def fn_wrapper(img):
        img_ = torch.tensor(img, device="cuda:0")
        img_ = img_.permute((0, 3, 1, 2))
        preds = predict_fn(img_)
        del img_
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        return preds.detach().cpu().numpy()

    original_image = np.transpose((image[0].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    explainer = lime_image.LimeImageExplainer()
    print(f"Running with batch size 1 ....")
    explanation = explainer.explain_instance(
        np.array(original_image),
        fn_wrapper,  # classification function
        top_labels=2,
        hide_color=0,
        num_samples=6000,
        batch_size=1,
    )  # nu
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False
    )
    img_boundry2 = mark_boundaries(temp, mask)
    fig, ax = plt.subplots()
    ax.imshow(img_boundry2)
    ax.axis("off")  # Optional: Hide axes for a cleaner look
    plt.tight_layout()
    mask = np.stack((mask,) * 3, axis=0)
    mask = np.expand_dims(mask, axis=0)
    return mask, fig


if __name__ == "__main__":
    # trash code to remove later, used to avoid lock

    set_seed()
    # Generate a random delay time between 0 and 200 seconds
    delay_time = random.uniform(0, 100)

    print(f"Delaying execution for {delay_time:.2f} seconds...")
    time.sleep(delay_time)
    print("Execution resumed.")
    args = parse_args()
    explainer_configs = {}

    json_dict = json.load(open(args.config_file))

    key = get_formatted_datetime()

    folds_pred = json_dict["dataset"]["folds_pred"]
    folds_pred_unique = (
        list(set(functools.reduce(lambda a, b: a + b, [folds for folds in folds_pred.values()]))) if folds_pred else []
    )

    df = pd.read_csv(json_dict["dataset"]["df"])
    df = df.loc[df["fold"].isin(folds_pred_unique)]
    data_module_config = json_dict["data_module_config"]

    if "normalization_values" not in data_module_config.keys():
        data_module_config["normalization_values"] = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]

    model_config = json_dict["model"]
    wandb_additional_config = {
        "data_module_config": data_module_config,
        "model_config": model_config,
        "dataset_config": json_dict["dataset"],
    }
    global model_name
    model_name = (
        model_config["architecture"].split("/")[-1]
        if isinstance(model_config, dict)
        else model_config.replace("/", "_", -1)
    )
    WANDB_RUN = handle_wandb(json_dict["wandb_config"], wandb_additional_config)

    fold_config = FoldConfig(
        df=df,
        folds_train=None,
        folds_val=None,
        folds_test=None,
        folds_pred=folds_pred,
        label_df=None,
    )

    data_module = PhysionetDataModule(fold_config, **data_module_config)
    lit_model = load_model(model_config, WANDB_RUN)

    os.makedirs(f"{args.output_dir}/models/{key}/preds")

    allowed_filepaths = []
    for pred_group, folds in fold_config.folds_pred.items():
        df_ = df[df["fold"].isin(folds)]
        if len(allowed_filepaths) > 0:
            df_ = df_[df_["path"].isin(allowed_filepaths)]
        else:
            df_ = df


        dataset = FromDfImageDataset(df_, data_module.data_dir, transform=data_module.val_test_transform)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=data_module.num_workers)

        logging.error(f"{explainer_configs=}")

        external_explainers = {"lime": get_lime_preds}

        explainer = Explainer(
            lit_model=lit_model,
            dataloader=dataloader,
            explainers=explainer_configs,
            external_explainers=external_explainers,
            folder_name=data_module_config['data_dir'].split("/")[-2]
        )
        if isinstance(lit_model, InferenceTree):

            def giraffe_predict(x):
                out = lit_model.predict(x)
                out = torch.stack([1 - out, out], dim=-1).reshape(-1, 2)
                return out

            explainer.iterate_explain(predict_fn=giraffe_predict)
        else:

            def lit_predict(x):
                lit_model.eval()
                out = lit_model(x)
                out = torch.stack([1 - out, out], dim=-1).reshape(-1, 2)
                return out

            explainer.iterate_explain(predict_fn=lit_predict)
        explainer.upload(WANDB_RUN)

    artifact = wandb.Artifact("preds" + WANDB_RUN.name, type="preds")
    artifact.add_dir(f"{args.output_dir}/models/{key}")
    WANDB_RUN.log_artifact(artifact)
    artifact.wait()
