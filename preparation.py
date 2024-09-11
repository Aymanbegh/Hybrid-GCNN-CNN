import os
import requests
import sys
from zipfile import ZipFile
from tqdm import tqdm
import gdown

import subprocess
import torch

# Définir la version de Torch comme variable d'environnement
os.environ['TORCH'] = torch.__version__
print(f"Torch version: {torch.__version__}")

# Installer torch-scatter
scatter_install_command = f"pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{os.environ['TORCH']}.html"
subprocess.run(scatter_install_command, shell=True, check=True)
print("torch-scatter installé avec succès.")

# Installer torch-sparse
sparse_install_command = f"pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{os.environ['TORCH']}.html"
subprocess.run(sparse_install_command, shell=True, check=True)
print("torch-sparse installé avec succès.")


def download_file(url, destination_path):
    """Télécharge un fichier depuis une URL en affichant une barre de progression."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Téléchargement avec une barre de progression
    with open(destination_path, 'wb') as file, tqdm(
            desc=destination_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    print(f"Files download and saved to {destination_path}.")


def unzip_file(zip_path, extract_to):
    """Décompresse un fichier ZIP dans le répertoire spécifié avec une barre de progression."""
    # Vérifie si le fichier ZIP existe
    if not os.path.exists(zip_path):
        print(f"Folder {zip_path} does not exist.")
        return

    # Crée le répertoire cible s'il n'existe pas
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print(f"Repository created : {extract_to}")

    # Ouverture du fichier ZIP
    with ZipFile(zip_path, 'r') as zip_ref:
        # Liste des fichiers dans le zip
        zip_file_list = zip_ref.namelist()

        # Barre de progression pour l'extraction
        with tqdm(total=len(zip_file_list), unit='file', desc="Extraction des fichiers") as progress_bar:
            for file in zip_file_list:
                zip_ref.extract(file, extract_to)
                progress_bar.update(1)  # Mise à jour de la barre de progression

        print(f"Content extract in : {extract_to}")


def download_weights_from_drive(file_id, output_dir, output_filename):
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Chemin complet du fichier de sortie
    output_path = os.path.join(output_dir, output_filename)

    # URL directe pour gdown à partir de l'ID du fichier Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"

    # Télécharger le fichier
    gdown.download(url, output_path, quiet=False)

    print(f"Fichier téléchargé et sauvegardé sous {output_path}")


def main(mode):
    # Définir les chemins
    current_directory = os.getcwd()
    base_dir = current_directory + '/data'
    model_path = current_directory + '/weights'
    json_file_url = 'https://www.l2ti.univ-paris13.fr/VSQuad/CD-COCO_ICIP2023_Challenge/train_annotations/train.json'
    if mode == 'all_normal':
        zip_file_url = 'http://images.cocodataset.org/zips/train2017.zip'
        zip_file_path = os.path.join(base_dir, 'train2017.zip')

    if mode == 'all_distorted':
        zip_file_url = 'https://www.l2ti.univ-paris13.fr/VSQuad/CD-COCO_ICIP2023_Challenge/train_val_data/train2017_distorted.zip'
        zip_file_path = os.path.join(base_dir, 'train2017_distorted.zip')

    json_file_path = os.path.join(base_dir, 'train.json')
    extracted_dir = os.path.join(base_dir)

    # Créer le répertoire 'data' si nécessaire
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Repository created: {base_dir}")

    # Télécharger le fichier JSON
    print(f'print json file : {json_file_path}')
    download_file(json_file_url, json_file_path)

    # Télécharger le fichier ZIP
    print(f'print coco file : {zip_file_path}')
    download_file(zip_file_url, zip_file_path)

    # Décompresser le fichier ZIP
    print(f'print dezip coco file : {extracted_dir}')
    unzip_file(zip_file_path, extracted_dir)

    # Exemple d'utilisation :
    file_id = '1bzO6M9YVeCr1XDJtFLpFiwOGIt9OTD_T'  # Remplacer par votre ID de fichier Google Drive
    output_dir = model_path  # Le dossier où vous souhaitez sauvegarder le fichier
    output_filename = 'model2_GIN_X2_1.pt'  # Le nom sous lequel sauvegarder le fichier

    # Télécharger le fichier
    download_weights_from_drive(file_id, output_dir, output_filename)

    # Exemple d'utilisation :
    file_id = '1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg'  # Remplacer par votre ID de fichier Google Drive
    output_dir = model_path  # Le dossier où vous souhaitez sauvegarder le fichier
    output_filename = 'yolact_im700_54_800000.pt'  # Le nom sous lequel sauvegarder le fichier

    # Download yolact model weight
    download_weights_from_drive(file_id, output_dir, output_filename)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py mode")
    else:
        mode_ = sys.argv[1]
        main(mode_)
