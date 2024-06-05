import torch
import random
import numpy as np
import gradio as gr
from PIL import Image
from collections import defaultdict
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerConfig, Mask2FormerModel

# Simply a color palattes for COCO, cityscapes and ADE datasets

def coco_panoptic_palette():
    return [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208), (255, 255, 128), (147, 211, 203),
               (150, 100, 100), (168, 171, 172), (146, 112, 198),
               (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
               (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
               (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
               (106, 154, 176),
               (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
               (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
               (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
               (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
               (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
               (146, 139, 141),
               (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
               (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
               (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
               (102, 102, 156), (250, 141, 255)]

def cityscapes_palette():
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def ade_palette():
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]]


# Initializing a Mask2Former facebook.mask2former-swin-small-coco-instance configuration
configuration = Mask2FormerConfig()

# Initializing a model (with random weights) from the facebook/mask2former-swin-small-coco=instance style config
model = Mask2FormerModel(configuration)

# Accessing the model configuration
configuration = model.config

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_processor(model_ckpt: str, device: str):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_ckpt).to(torch.device(device))
    model.eval()
    image_processor = MaskFormerImageProcessor.from_pretrained(model_ckpt)
    return model, image_processor

# Choose what model to download from Huggingface for SEMANTIC or PANOPTIC (everythink visible) task:
def load_default_ckpt(segmentation_task: str):
    if segmentation_task == "semantic":
        # default_pretrained_ckpt = "facebook/mask2former-swin-base-ade-semantic"
        default_pretrained_ckpt = "/root/autodl-tmp/mask2former-swin-base-ade-semantic"
    elif segmentation_task == "panoptic":
        # default_pretrained_ckpt = "facebook/mask2former-swin-large-coco-panoptic"
        default_pretrained_ckpt = "/root/autodl-tmp/mask2former-swin-large-coco-panoptic"
    return default_pretrained_ckpt

def draw_panoptic_segmentation(predicted_segmentation_map, seg_info, image):
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    for res in seg_info:
        res['category_id'] = res.pop('label_id')
        pred_class = res['category_id']
        isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
        res['isthing']= bool(isthing)
        
    visualizer = Visualizer(np.array(image)[:,:,::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    out = visualizer.draw_panoptic_seg_predictions(
        predicted_segmentation_map.cpu(), seg_info, alpha=0.5
    )
    output_img = Image.fromarray(out.get_image())
    return output_img

def draw_semantic_segmentation(segmentation_map, image, palette):
    color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_segmentation_map[segmentation_map - 1 == label, :]= color
    # Convert to BGR
    ground_truth_color_seg = color_segmentation_map[..., ::-1]
    img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
    img = img.astype(np.uint8)
    return img

# Main sequence of code:
def predict_masks(input_img_path: str, segmentation_task: str):
    
    # load model checkpoints and image processor for it:
    default_pretrained_ckpt = load_default_ckpt(segmentation_task)
    model, image_processor = load_model_and_processor(default_pretrained_ckpt, device)
    
    ## pass input image through processor
    image = Image.open(input_img_path)
    inputs = image_processor(images=image, return_tensors="pt").to(torch.device(device))
    
    ## pass inputs to model for prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    ## pass outputs to processor for POSTPROCESSING:
    if segmentation_task == "semantic":
        result = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_segmentation_map = result.cpu().numpy()
        palette = ade_palette()
        output_result = draw_semantic_segmentation(predicted_segmentation_map, image, palette)
    else:
        result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_segmentation_map = result["segmentation"]
        seg_info = result["segments_info"]
        output_result = draw_panoptic_segmentation(predicted_segmentation_map, seg_info, image)
    return output_result


with gr.Blocks() as Mask2Former_GUI:
    
    gr.Markdown("# **<p align='center'>Mask2Former for Panoptic and Semantic Segmentation</p>**")
    gr.Markdown("<p align='center'> A Mask2Former GUI to run on RTX3090_24GB </p>")
    
    with gr.Group():
        with gr.Row():
            choose_segmentation = gr.Dropdown(["semantic", "panoptic"], value="panoptic", label="Panoptic or Semantic Segmentation")
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type='filepath', label="Place your image here:", show_label=True)
                with gr.Column(scale=3):
                    output_mask = gr.Image(label="Mask2Former Output:", show_label=True)
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                start_run = gr.Button("Run Mask2Former-Inference")
            with gr.Column(scale=2):
                gr.Markdown('\n my Mask2Former models are \
                facebook/mask2former-swin-base-ade-semantic \
                and facebook/mask2former-swin-large-coco-panoptic.')
    
    start_run.click(predict_masks, inputs=[input_image, choose_segmentation], outputs=output_mask)

if __name__ == "__main__":
    Mask2Former_GUI.launch(share=False, debug=False)
