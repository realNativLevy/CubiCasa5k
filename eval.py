import numpy as np
import logging
import argparse
import torch
from datetime import datetime
from torch.utils import data
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import DictToTensor, Compose
from floortrans.metrics import get_evaluation_tensors, runningScore
from tqdm import tqdm
import matplotlib
import os
matplotlib.use('Agg')  # Change from TkAgg to Agg
from floortrans.plotting import segmentation_plot
import matplotlib.pyplot as plt
import cv2

room_cls = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
icon_cls = ["Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"]


def print_res(name, res, cls_names, logger):
    basic_res = res[0]
    class_res = res[1]

    basic_names = ''
    basic_values = name
    basic_res_list = ["Overall Acc", "Mean Acc", "Mean IoU", "FreqW Acc"]
    for key in basic_res_list:
        basic_names += ' & ' + key
        val = round(basic_res[key] * 100, 1)
        basic_values += ' & ' + str(val)

    logger.info(basic_names)
    logger.info(basic_values)

    basic_res_list = ["IoU", "Acc"]
    logger.info("IoU & Acc")
    for i, name in enumerate(cls_names):
        iou = class_res['Class IoU'][str(i)]
        acc = class_res['Class Acc'][str(i)]
        iou = round(iou * 100, 1)
        acc = round(acc * 100, 1)
        logger.info(name + " & " + str(iou) + " & " + str(acc) + " \\\\ \\hline")


def evaluate(args, log_dir, logger):
    normal_set = FloorplanSVG(args.data_path, 'test.txt', format='txt', augmentations=Compose([DictToTensor()]))
    data_loader = data.DataLoader(normal_set, batch_size=1, num_workers=0)

    # Setup device - use CPU for consistency
    device = torch.device('cpu')
    
    # Load checkpoint with correct device mapping
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Setup Model
    model = get_model(args.arch, 51)
    n_classes = args.n_classes
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)

    # Create output directory for visualizations
    vis_dir = os.path.join(log_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    score_seg_room = runningScore(12)
    score_seg_icon = runningScore(11)
    score_pol_seg_room = runningScore(12)
    score_pol_seg_icon = runningScore(11)
    
    with torch.no_grad():
        for count, val in tqdm(enumerate(data_loader), total=len(data_loader),
                               ncols=80, leave=False):
            logger.info(count)
            
            # Get predictions
            things = get_evaluation_tensors(val, model, split, logger, rotate=True)
            label, segmentation, pol_segmentation = things

            # Get original image if available in val
            original_img = val.get('image', None)

            # Update scores
            score_seg_room.update(label[0], segmentation[0])
            score_seg_icon.update(label[1], segmentation[1])
            score_pol_seg_room.update(label[0], pol_segmentation[0])
            score_pol_seg_icon.update(label[1], pol_segmentation[1])

    generateWallMeshPlotImage(pol_segmentation, vis_dir, count)
    generateFloorplanVariations(segmentation, pol_segmentation, vis_dir, count)
    generateWallsPlotImage(pol_segmentation)
    
    # Print metrics
    print_res("Room segmentation", score_seg_room.get_scores(), room_cls, logger)
    print_res("Room polygon segmentation", score_pol_seg_room.get_scores(), room_cls, logger)
    print_res("Icon segmentation", score_seg_icon.get_scores(), icon_cls, logger)
    print_res("Icon polygon segmentation", score_pol_seg_icon.get_scores(), icon_cls, logger)

    logger.info(f"Visualizations saved in: {vis_dir}")

def generateWallMeshPlotImage(pol_segmentation, vis_dir, count):
    # Create a 3D visualization of wall mesh
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract wall contours
    walls_only = np.zeros_like(pol_segmentation[0])
    walls_only[pol_segmentation[0] == 2] = 1  # Copy only wall class
    walls_only = (walls_only * 255).astype(np.uint8)
    contours, _ = cv2.findContours(walls_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set wall height
    wall_height = 3.0  # 3 meters tall walls
    
    # Plot each wall contour as a 3D surface
    for contour in contours:
        contour = contour.squeeze()
        if len(contour.shape) >= 2:
            # Create wall surface points
            x = contour[:, 0]
            y = contour[:, 1] 
            z = np.zeros_like(x)
            
            # Plot vertical walls
            for i in range(len(x)-1):
                wall_x = [x[i], x[i], x[i+1], x[i+1]]
                wall_y = [y[i], y[i], y[i+1], y[i+1]]
                wall_z = [0, wall_height, wall_height, 0]
                
                # Create wall surface
                ax.plot_trisurf(wall_x, wall_y, wall_z, color='gray', alpha=0.7)

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,0.3])
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('3D Wall Mesh')

    # Save figure
    plt.savefig(os.path.join(vis_dir, f'wall_mesh_{count}.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    
def generateFloorplanVariations(segmentation, pol_segmentation, vis_dir, count):
    # Create figure with subplots for regular visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Plot room segmentation
    plt.sca(axes[0,0])
    axes[0,0].set_title('Room Segmentation')
    segmentation_plot(segmentation[0])
    
    # Plot room polygon segmentation
    plt.sca(axes[0,1])
    axes[0,1].set_title('Room Polygon Segmentation')
    segmentation_plot(pol_segmentation[0])
    
    # Plot icon segmentation
    plt.sca(axes[1,0])
    axes[1,0].set_title('Icon Segmentation')
    segmentation_plot(segmentation[1])
    
    # Plot icon polygon segmentation
    plt.sca(axes[1,1])
    axes[1,1].set_title('Icon Polygon Segmentation')
    segmentation_plot(pol_segmentation[1])

    # Save regular visualization
    plt.savefig(os.path.join(vis_dir, f'floorplan_{count}.png'))
    plt.close()


def generateWallsPlotImage(pol_segmentation, vis_dir, count):
    # Create walls-only visualization with contours
    walls_only = np.zeros_like(pol_segmentation[0])
    walls_only[pol_segmentation[0] == 2] = 1  # Copy only wall class
    
    # Convert to uint8 for contour detection
    walls_only = (walls_only * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(walls_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create clean figure for walls
    plt.figure(figsize=(10, 10))
    plt.title('Walls Only')
    plt.axis('off')
    
    # Plot each contour as a line
    for contour in contours:
        # Reshape contour for plotting
        contour = contour.squeeze()
        if len(contour.shape) >= 2:  # Check if valid contour
            # Plot the contour lines
            plt.plot(contour[:, 0], contour[:, 1], 'k-', linewidth=1)
    
    # Set equal aspect ratio and invert y-axis for proper wall display
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    
    plt.savefig(os.path.join(vis_dir, f'walls_only_{count}.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def evaluate_all_floorplans(args, logger):
    """Evaluate all floorplans in the high quality architectural dataset"""
    base_path = 'data/cubicasa5k/high_quality_architectural'
    
    # Get all subdirectories
    subdirs = []
    for entry in os.scandir(base_path):
        if entry.is_dir():
            subdirs.append(entry.path)
    
    print(f"Found {len(subdirs)} floorplans to process")
    
    for subdir in tqdm(subdirs, desc="Processing floorplans"):
        # Create a temporary test.txt file in the subdir with just this floorplan
        floorplan_name = os.path.basename(subdir)
        temp_test_file = os.path.join(subdir, 'test.txt')
        with open(temp_test_file, 'w') as f:
            f.write(f"high_quality_architectural/{floorplan_name}")
        
        # Update args for this specific floorplan
        args.log_path = subdir  # Save outputs in the same directory as input
        
        # Create timestamp directory inside the floorplan directory
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        log_dir = os.path.join(subdir, 'evaluation_' + time_stamp)
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            # Evaluate this floorplan
            evaluate(args, log_dir, logger)
            print(f"Successfully processed {floorplan_name}")
        except Exception as e:
            print(f"Error processing {floorplan_name}: {str(e)}")
        finally:
            # Clean up temporary test file
            if os.path.exists(temp_test_file):
                os.remove(temp_test_file)

# if __name__ == '__main__1':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Settings for evaluation')
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use [\'hg_furukawa_original, segnet etc\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of the epochs')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file .pkl')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')
    parser.add_argument('--output-path', nargs='?', type=str, default='runs_cubi_out/',
                        help='Path to output directory')

    args = parser.parse_args()
    print("Instance of args: ", args)
    log_dir = args.log_path + '/' + time_stamp + '/'

    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True) 

    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/eval.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    evaluate(args, log_dir, logger)


if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Settings for evaluation')
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use [\'hg_furukawa_original, segnet etc\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of the epochs')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file .pkl')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')
    parser.add_argument('--output-path', nargs='?', type=str, default='runs_cubi_out/',
                        help='Path to output directory')

    args = parser.parse_args()
    print("Instance of args: ", args)
    log_dir = args.log_path + '/' + time_stamp + '/'

    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True) 

    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/eval.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    print("Evaluating all floorplans")
    evaluate_all_floorplans(args,logger)