import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
import utils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
        in_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_dim, num_classes)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        return self.resnet(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # Aim for an mAP of around 0.8 (80%) in 50 epochs.
    ##################################################################
    args = ARGS(
        epochs=50,
        inp_size=256,
        use_cuda=True,
        val_every=70,
        lr=0.0001,
        batch_size=32,
        step_size=15,
        gamma=0.1,
        log_dir="runs/q2_best",
        save_at_end=True
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)

    # Q2 t-SNE plot guidance:
    # Use this fine-tuned Q2 classifier in evaluation mode, not a fresh
    # ImageNet model. Extract the average pooling output immediately before
    # the final fully connected layer (resnet.avgpool), and flatten it to
    # one 512-dimensional feature vector per image for ResNet-18.
    # Use the 20 base PASCAL VOC classes (VOCDataset.CLASS_NAMES) in the
    # legend, rather than creating entries for multi-label combinations.

    feature_extractor = nn.Sequential(*list(model.resnet.children())[:-1])

    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.eval()

    all_features = []
    all_targets = []

    model.eval()

    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    with torch.no_grad():
        for data, target, wgt in test_loader:
            data = data.to(args.device)

            # Output shape: [B, 512, 1, 1]
            feat = feature_extractor(data)

            # Flatten to [B, 512]
            feat = feat.flatten(1)

            all_features.append(feat.cpu())
            all_targets.append(target.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    print("Feature shape:", features.shape)
    print("Target shape:", targets.shape)

    rng = np.random.default_rng(42)

    num_samples = min(1000, len(features))

    indices = rng.choice(
        len(features),
        size=num_samples,
        replace=False
    )

    features = features[indices]
    targets = targets[indices]

    print("Sampled feature shape:", features.shape)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    features_2d = tsne.fit_transform(features)

    print("t-SNE shape:", features_2d.shape)

    class_names = VOCDataset.CLASS_NAMES
    num_classes = len(class_names)

    cmap = plt.get_cmap("tab20")

    class_colors = np.array([
        cmap(i)[:3]
        for i in range(num_classes)
    ])

    point_colors = []

    for target in targets:

        active_classes = np.where(target > 0.5)[0]

        if len(active_classes) > 0:
            color = class_colors[active_classes].mean(axis=0)
        else:
            color = np.array([0.5, 0.5, 0.5])

        point_colors.append(color)

    point_colors = np.array(point_colors)

    plt.figure(figsize=(12, 9))

    plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=point_colors,
        s=20,
        alpha=0.75,
        edgecolors="none"
    )

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE of Fine-tuned ResNet-18 Features")

    legend_handles = []

    for i, class_name in enumerate(class_names):

        handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=class_colors[i],
            markeredgecolor="none",
            markersize=8,
            label=class_name
        )

        legend_handles.append(handle)


    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(
        "tsne_resnet18.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()