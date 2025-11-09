import os
import wandb

from giraffe.inferece_tree import InferenceTree
from giraffe.tree import Tree

from src.lit_models.guangzhou_models import GuangzhouLitModel

class GiraffeHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_relevant_runs(project_name, tree_arch):
        api = wandb.Api()
        project = api.project(project_name)
        runs = api.runs("/".join(project.path))
        print(f"wandb runs: {runs}")
        relevant_runs = []
        tree = Tree.load_tree_architecture(tree_arch)
        unique_ids = [node_id.replace(".pt", "") for node_id in tree.get_unique_value_node_ids()]
        for run in runs:
            try:
                if run.name in unique_ids:
                    relevant_runs.append(run)
            except Exception as e:
                print(e)
        return relevant_runs

    @staticmethod
    def pred_function(run):
        path = None
        for artifact in run.logged_artifacts():
            if artifact.type == "model":
                path = artifact.download()

        def func(x):
            model = GuangzhouLitModel.load_from_checkpoint(os.path.join(path, "model.ckpt"))
            model.zero_grad()
            model.eval()
            return model(x)

        return func

    @staticmethod
    def build_pred_func_dict(tree_architecture_path, project_name):
        relevant_runs = GiraffeHandler.get_relevant_runs(project_name, tree_architecture_path)
        print(f"Relevent_runs: {relevant_runs}")
        pred_func_dict = {}
        for run in relevant_runs:
            print(f"Relevant prediction function: {GiraffeHandler.pred_function(run)}")
            pred_func_dict[run.name + ".pt"] = GiraffeHandler.pred_function(run)
        return pred_func_dict