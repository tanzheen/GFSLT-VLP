import torch 
import gzip 
import pickle 
from PIL import Image 
import os 
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import sys 

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset_file(filename):
        with gzip.open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object
    all_paths = []
    train_labels = 'E:\GFSLT-VLP\data\Phonexi-2014T/labels.train'
    train_labels = load_dataset_file(train_labels)
    val_labels = 'E:\GFSLT-VLP\data\Phonexi-2014T/labels.dev'
    val_labels = load_dataset_file(val_labels)
    test_labels = 'E:\GFSLT-VLP\data\Phonexi-2014T/labels.test'
    test_labels = load_dataset_file(test_labels)

    for value in train_labels.values(): 
        path_lst = value['imgs_path']
        all_paths.extend(path_lst)

    for value in val_labels.values(): 
        path_lst = value['imgs_path']
        all_paths.extend(path_lst)

    for value in test_labels.values(): 
        path_lst = value['imgs_path']
        all_paths.extend(path_lst)

    #print(all_paths)

    # Open up the label files and collate all the labels

    ## use openai vit clip 

    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")
    model.to(device)
    model.eval()
    print(f"number of images: {len(all_paths)}")

    def tokenize_and_save(img_path):
        img_path = os.path.join('E:\PHOENIX-2014-T-release-v3\PHOENIX-2014-T/features/fullFrame-210x260px/', img_path)
        print(img_path)
        original_image = Image.open(img_path)
        processed_image = processor(images=[original_image], return_tensors="pt")
        with torch.no_grad():
            spatial_feats = model.get_image_features(pixel_values=processed_image.pixel_values.to(device))
        spatial_feats= spatial_feats.squeeze() 
        print(spatial_feats.size())
        # save the encoded tokens 
        new_file_path = os.path.splitext(img_path)[0] + ".pth"
        torch.save(spatial_feats, new_file_path)
    print("Tokenizing! ")
    for img_path in all_paths: 
        tokenize_and_save(img_path)



if __name__ == "__main__":
 
    sys.path.append("..")
    print(torch.__version__)
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
        print(torch.backends.cuda.flash_sdp_enabled())
    
    torch.cuda.empty_cache()
    main()