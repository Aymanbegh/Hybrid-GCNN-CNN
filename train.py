import argparse
import random
from collections import defaultdict
import numpy as np
import networkx as nx
import math
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os.path as osp

# Import model configuration
from gin import GIN
from gcn import GCN
from ginlaf import LAFNet
from COCOParser import COCOParser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Hybrid-GCNN-CNN CD-COCO Training')
    parser.add_argument('--dataset', default="./data/train.json", type=str,
                        help='Specify the path to the CD-COCO groundtruth annotation.')
    parser.add_argument('--gcn_model', default=None, type=str,
                        help='Specify the model weights.')
    parser.add_argument('--model', default="GCN", type=str,
                        help='Specify the model type: GCN , GIN or GINLAF.')
    parser.add_argument('--file_res', default="./datat/trainings.txt", type=str,
                        help='Specify the path for the scene type groundtruth.')
    parser.add_argument('--nb_label', default=1, type=int,
                        help='Specify the accepted minimal number of different labels in a scene.')
    parser.add_argument('--beta', default=0.1, type=float,
                        help='Specify the beta value for the distance threshold.')
    parser.add_argument('--hidden', default=512, type=int,
                        help='Specify the hidden layers size for the model.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Specify the number of epoch for training.')
    parser.add_argument('--batch_sizes', default=64, type=int,
                        help='Specify the nbatch size parameter for training.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Specify the learning rate for training.')

    global args
    args = parser.parse_args(argv)


def Prepare_data(coco, ratio, content, nb_images, coef, coef_):
    train_set = nb_images * coef
    valid_set = nb_images * coef_
    total_images = len(coco.get_imgIds())  # total number of images

    # get images index
    img_ids = coco.get_imgIds()
    dataset = []

    for im in range(nb_images):
        # Select images
        selected_img_ids = img_ids[im]
        ann_ids = coco.get_annIds(selected_img_ids)
        im_licenses = coco.get_imgLicenses(selected_img_ids)
        name = str(selected_img_ids).zfill(12)
        Complet_name = name + ".jpg"
        label = []
        position_ = []
        class_ = []
        # Get annotations of image number img
        annotations = coco.load_anns(ann_ids)
        length = len(annotations)

        for ann in annotations:
            class_id = ann["category_id"]
            class_.append(class_id)
        unique = set(class_)
        length_ = len(unique)

        if length_ > 2:
            # Extract information from img line of the "training.txt" file
            currentlines = content[im].split(",")
            label.append(int(currentlines[1]))

            # Create Graph G
            G = nx.Graph()

            # Create nodes and edges list
            nodes = []
            edges = []
            edges_ = []
            sets = []
            if im <= train_set:
                tm = True
                vm = False
                tsm = False

            if im > train_set and im <= train_set + valid_set:
                tm = False
                vm = True
                tsm = False

            if im > train_set + valid_set:
                tm = False
                vm = False
                tsm = True

            train_mask = tm
            val_mask = vm
            test_mask = tsm

            information = []
            j = 0
            for ann in annotations:
                # Extract the bounding boxe information
                bbox = ann['bbox']
                x, y, w, h = [int(b) for b in bbox]
                pos = np.array([[x, y]])
                posi = [x, y]
                if j == 0:
                    position = pos
                else:
                    position = np.concatenate((position, pos), axis=0)
                # Compute the object diagonal
                diagonal = float(math.sqrt(w * w + h * h))

                # Extract the object class information
                class_id = ann["category_id"]
                class_name = coco.load_cats(class_id)[0]["name"]
                # if im == 0 :
                #   print(class_name)
                license = coco.get_imgLicenses(selected_img_ids)[0]["name"]
                # Add node to the Graph G
                G.add_node(im, classe=class_id, size=diagonal)
                color = nx.get_node_attributes(G, 'classe')
                # nodes.append({"id": j, "attributes": {"class": float(class_id), "size": diagonal}})
                info = [x, y, w, h, class_id, class_name, diagonal]
                position_.append(posi)
                information.append(
                    info)  # print("VAL: {} Classe label : {} and node: {}".format(im,class_name,color[im]))
                j = j + 1

            # print(position_)
            min_value = min(position_)
            min_index = position_.index(min_value)
            information = sorted(information)

            for i in range(0, length - 1):
                for j in range((i + 1), length):
                    # Computhe distance between objects to define corresponding weight for specific edge
                    distance = (
                        math.sqrt((position[i][0] - position[j][0]) ** 2 + (position[i][1] - position[j][1]) ** 2))
                    # print(distance)
                    G.add_edge(i, j, weight=distance)
                    edges.append({"source": i, "target": j, "attributes": {"weight": distance}})

            for i in range(0, length):
                nodes.append({"id": i, "attributes": {"class": float(information[i][4]), "size": information[i][6]}})
                distances = []
                if i < length:
                    for j in range((i + 1), length):
                        # Computhe distance between objects to define corresponding weight for specific edge
                        dist = (math.sqrt((information[i][0] - information[j][0]) ** 2 + (
                                    information[i][1] - information[j][1]) ** 2))
                        distances.append(dist)
                        # print("value i: {} and j: {} distance: {}".format(i,j, distances))
                else:
                    dist = 0
                    distance.append(dist)
                if (len(distances) > 0):
                    min_value = min(distances)
                    min_index = distances.index(min_value)
                else:
                    min_value = 1000
                    min_index = 0
                if i < (length - 1):
                    idx = min_index + 1 + i
                    edges_.append({"source": i, "target": idx, "attributes": {"weight": min_value}})
                    # print(edges_)
                    for j in range(len(distances)):
                        if ((distances[j] <= (min_value + ratio * min_value)) and j != min_index):
                            if j < i:
                                idx = j
                            else:
                                idx = j + 1
                            edges_.append({"source": i, "target": idx, "attributes": {"weight": distances[j]}})

            node_attributes = []
            for node in nodes:
                node_attributes.append([node["attributes"][key] for key in node["attributes"]])

            s = []
            t = []
            edge_index = []
            edge_attributes = []
            # y = []

            for edge in edges_:
                source = int(edge["source"])
                target = int(edge["target"])
                # print("Source: {} and Target: {}".format(source,target))
                # edge_index.append([source, target])
                s.append(source)
                t.append(target)
                edge_attributes.append([edge["attributes"][key] for key in edge["attributes"]])

            edge_index = [s, t]

            # Tensorised
            node_attributes = torch.Tensor(node_attributes)
            position_ = torch.Tensor(position_)
            edge_index = torch.Tensor(edge_index)
            edge_index = edge_index.to(torch.long)
            label = torch.Tensor(label)
            label = label.to(torch.long)
            edge_attributes = torch.Tensor(edge_attributes)
            edge_attributes = edge_attributes.to(torch.long)

            # Global Network parameters
            num_nodes = len(nodes)
            num_node_features = 2
            num_edge_features = 1
            num_classes = 2

            datas = Data(x=node_attributes, edge_index=edge_index, edge_attr=edge_attributes, y=label, pos=position_,
                         train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
            dataset.append(datas)
    return dataset


def Split_dataset(dataset, coef, coef_, batch_sizes):
    # Create training, validation, and test sets
    train_dataset = dataset[:int(len(dataset) * coef)]
    val_dataset = dataset[int(len(dataset) * coef):int(len(dataset) * (coef + coef_))]
    test_dataset = dataset[int(len(dataset) * (coef + coef_)):]

    print(f'Training set   = {len(train_dataset)} graphs')
    print(f'Validation set = {len(val_dataset)} graphs')
    print(f'Test set       = {len(test_dataset)} graphs')

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=True)

    print('\nTest loader:')
    for i, batch in enumerate(test_loader):
        print(f' - Batch {i}: {batch}')

    return train_loader, val_loader, test_loader


def train(model, loader, lr, epochs, device, val_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # epochs = 7

    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0
        nb = 0
        # Train on batches
        for data in loader:
            model, data = model.to(device), data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()

            # Validation
            val_loss, val_acc = test(model, val_loader, device)
        print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
        test_loss, test_acc = test(model, test_loader, device)
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    return model

@torch.no_grad()
def testf(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        model, data = model.to(device), data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
    return loss, acc

def test(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        model, data = model.to(device), data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()



if __name__ == '__main__':
    parse_args()
    print('Loading parameters!!!')
    # Model parameters
    num_node_features = 2
    num_edge_features = 1
    hidden_channels = args.hidden
    num_classes = 2
    ratio = args.beta

    #dataset parameter
    nb_images = 90000
    coef = 0.8
    coef_ = (1 - coef) / 2

    print("Minimum number of label classes: {} and beta value: {}".format(args.nb_label, args.beta))

    # Training parameter
    lr = args.lr
    batch_sizes = args.batch_sizes
    epoch = args.epoch
    # open the scene type groundtruth file
    file = open(args.file_res)
    # read the content of the file opened
    content = file.readlines()

    print('Prepare dataset!')
    # Get the cd-coco annotations
    coco_annotations_file = args.dataset
    coco_images_dir = ".data/"
    coco = COCOParser(coco_annotations_file, coco_images_dir)
    dataset = Prepare_data(coco, ratio, content, nb_images, coef, coef_)
    train_loader, val_loader, test_loader = Split_dataset(dataset, coef, coef_, batch_sizes)
    print('Dataset launched!')

    #Create the Graph model
    if args.model == 'GIN':
        print("GIN PROCESS")
        model_gcn = GIN(num_node_features,hidden_channels,num_classes)

    if args.model == "GCN":
        print("GCN PROCESS")
        model_gcn = GCN(num_node_features, hidden_channels, num_classes)

    if args.model == 'GINLAF':
        print("GINLAF PROCESS")
        model_gcn = LAFNet(num_node_features, hidden_channels, num_classes)
    print("Graph MODEL created")

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    model = train(model_gcn, train_loader, lr, epoch, device, val_loader, test_loader)
    test_loss, test_acc = testf(model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
    print()
    torch.save(model.state_dict(), f'model_{args.model}_lab{args.beta}_hid{args.hidden}.pth')



