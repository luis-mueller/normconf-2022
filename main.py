import torch
import tqdm
import wandb
import hydra
from torchmetrics.functional import accuracy
from torch_geometric.nn.models.basic_gnn import GAT, GCN
from torch_geometric.datasets import TUDataset
from torch_geometric.seed import seed_everything
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean


@hydra.main(version_base=None, config_name="config", config_path="./")
def main(cfg):
    seed_everything(cfg.hparams.seed)

    # No need to manually download the datasets, since torch_geometric already has
    # wrappers for them.
    if cfg.hparams.Dataset == "Enzymes":
        dataset = TUDataset(root="data/ENZYMES", name="ENZYMES", use_node_attr=True)
        split = [500, 100]
        dim_in, dim_hidden, dim_out = 21, 64, 6
    elif cfg.hparams.Dataset == "Proteins":
        dataset = TUDataset(root="data/PROTEINS", name="PROTEINS", use_node_attr=True)
        split = [1000, 113]
        dim_in, dim_hidden, dim_out = 4, 64, 2
    else:
        raise ValueError(f"Dataset not supported: {cfg.hparams.Dataset}")

    if cfg.hparams.Model == "Graph Convolution":
        model = GCN(dim_in, dim_hidden, 2, jk="last")
    elif cfg.hparams.Model == "Graph Attention":
        model = GAT(dim_in, dim_hidden, 2, jk="last")
    else:
        raise ValueError(f"Model not supported: {cfg.hparams.Model}")

    predictor = torch.nn.Sequential(
        torch.nn.Linear(dim_hidden, dim_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_hidden, dim_out),
    )

    train_data, test_data = random_split(dataset, split)
    train_loader = DataLoader(train_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.hparams.Dataset}-{cfg.hparams.Model}-{cfg.hparams.seed}",
        
        # 🤔 All hyper-params of the experiment are already grouped under
        # hparams and can later be used to build plots and tables.
        config=dict(cfg.hparams),

        # Not strictly necessary, but helps to keep an overview on the cloud
        tags=[cfg.hparams.Dataset, cfg.hparams.Model],
    )

    print("Training:")
    iteration = 0
    for _ in (pbar := tqdm.tqdm(range(250))):
        for batch in train_loader:
            opt.zero_grad()
            x = model(batch.x, batch.edge_index)
            x = scatter_mean(x, batch.batch, dim=0)
            preds = predictor(x)
            loss: torch.Tensor = loss_fn(preds, batch.y)
            loss.backward()
            opt.step()
            acc = accuracy(preds, batch.y, "multiclass", num_classes=dim_out)
            wandb.log(
                {"Iteration": iteration, "Loss (train)": loss, "Accuracy (train)": acc}
            )
            pbar.set_description(f"Loss: {float(loss):.3f}, Accuracy: {float(acc):.3f}")
            iteration += 1

    print("Test:")
    model.eval()
    for _ in (pbar := tqdm.tqdm(range(50))):
        for batch in test_loader:
            x = model(batch.x, batch.edge_index)
            x = scatter_mean(x, batch.batch, dim=0)
            preds = predictor(x)
            acc = accuracy(preds, batch.y, "multiclass", num_classes=dim_out)
            wandb.log({"Accuracy (test)": acc})
            pbar.set_description(f"Accuracy: {float(acc):.3f}")
    wandb.finish()


if __name__ == "__main__":
    main()
