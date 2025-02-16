import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D   # Import this to register the '3d' projection
import numpy as np
import logging
import argparse
from tensorboardX import SummaryWriter
import torch
from datetime import datetime
from torch.utils import data
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import DictToTensor, Compose
from floortrans.metrics import get_evaluation_tensors, runningScore, polygons_to_tensor
from tqdm import tqdm
import os
from floortrans.plotting import segmentation_plot
import matplotlib.pyplot as plt
import cv2
import shutil
from floortrans import post_prosessing

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


def original_evaluate(args, log_dir, writer, logger):

    normal_set = FloorplanSVG(args.data_path, 'test.txt', format='lmdb', lmdb_folder='cubi_lmdb/', augmentations=Compose([DictToTensor()]))
    data_loader = data.DataLoader(normal_set, batch_size=1, num_workers=0)

    checkpoint = torch.load(args.weights)
    # Setup Model
    model = get_model(args.arch, 51)
    n_classes = args.n_classes
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.cuda()

    score_seg_room = runningScore(12)
    score_seg_icon = runningScore(11)
    score_pol_seg_room = runningScore(12)
    score_pol_seg_icon = runningScore(11)
    with torch.no_grad():
        for count, val in tqdm(enumerate(data_loader), total=len(data_loader),
                               ncols=80, leave=False):
            logger.info(count)
            things = get_evaluation_tensors(val, model, split, logger, rotate=True)

            label, segmentation, pol_segmentation = things

            score_seg_room.update(label[0], segmentation[0])
            score_seg_icon.update(label[1], segmentation[1])

            score_pol_seg_room.update(label[0], pol_segmentation[0])
            score_pol_seg_icon.update(label[1], pol_segmentation[1])

    print_res("Room segmentation", score_seg_room.get_scores(), room_cls, logger)
    print_res("Room polygon segmentation", score_pol_seg_room.get_scores(), room_cls, logger)
    print_res("Icon segmentation", score_seg_icon.get_scores(), icon_cls, logger)
    print_res("Icon polygon segmentation", score_pol_seg_icon.get_scores(), icon_cls, logger)


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
    print(f"Visualizations will be saved in: {vis_dir}")
    score_seg_room = runningScore(12)
    score_seg_icon = runningScore(11)
    score_pol_seg_room = runningScore(12)
    score_pol_seg_icon = runningScore(11)
    
    with torch.no_grad():
        for count, val in tqdm(enumerate(data_loader), total=len(data_loader),
                               ncols=80, leave=False):
            logger.info(count)
            print(f"Processing floorplan {count}")
            
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
            
            print(f"Generating visualizations for count: {count}")
            generateWallMeshPlotImage(pol_segmentation, vis_dir, count)
            generateFloorplanVariations(segmentation, pol_segmentation, vis_dir, count)
            generateWallsPlotImage(pol_segmentation, vis_dir, count)

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

                # Ensure we have at least 3 unique points for triangulation
                if len(np.unique(wall_x)) < 3 or len(np.unique(wall_y)) < 3:
                    continue

                # Create wall surface
                ax.plot_trisurf(wall_x, wall_y, wall_z, color='gray', alpha=0.7)

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,0.3])
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('3D Wall Mesh')

    print(f"Full path: {os.path.join(vis_dir, f'wall_mesh_{count}.png')}")
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

    print(f"Full path: {os.path.join(vis_dir, f'floorplan_{count}.png')}")

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
    print(f"Full path: {os.path.join(vis_dir, f'walls_only_{count}.png')}")
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

        floorplan_name = os.path.basename(subdir)
      
        # Update args for this specific floorplan
        args.log_path = subdir  # Save outputs in the same directory as input
        
        # Create timestamp directory inside the floorplan directory
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        # Remove any existing evaluation_ folders
        for item in os.listdir(subdir):
            if item.startswith('evaluation_'):
                try:
                    shutil.rmtree(os.path.join(subdir, item))
                    print(f"Removed {item}")
                except PermissionError:
                    print(f"Permission denied when trying to remove {item}")
                except FileNotFoundError:
                    print(f"Directory {item} does not exist")
                except Exception as e:
                    print(f"Error removing {item}: {str(e)}")
                    
        log_dir = os.path.join(subdir, 'evaluation_' + time_stamp)
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"Processing {floorplan_name}")
        print(f"Log directory: {log_dir}")
        print(f"Args: {args}")
        try:
            # Evaluate this floorplan
            evaluate(args, log_dir, logger)
            print(f"Successfully processed {floorplan_name}")
        except Exception as e:
            print(f"Error processing {floorplan_name}: {str(e)}")         

if __name__ == '__main__':
    print("Evaluating single floorplan")
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
    os.makedirs(log_dir, exist_ok=True)  # Create log directory
    
    # Setup device and model
    device = torch.device('cpu')
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Load and preprocess image
    fplan = cv2.imread('F1_original.png')
    if fplan is None:
        raise ValueError("Failed to load F1_original.png")
        
    # Resize image if needed (adjust size as needed)
    fplan = cv2.resize(fplan, (512, 512))
    
    # Preprocess image
    fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)
    fplan = np.moveaxis(fplan, -1, 0)
    fplan = 2 * (fplan / 255.0) - 1
    image = torch.FloatTensor(fplan).unsqueeze(0)
    
    # Setup Model
    model = get_model(args.arch, 51)
    n_classes = args.n_classes
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()  # Set to evaluation mode
    
    # Get predictions
    with torch.no_grad():
        pred = model(image)
        
    # Process predictions
    segmentation = pred[:, :split[0]]  # Room segmentation
    icon_pred = pred[:, split[0]:split[0]+split[1]]  # Icon segmentation
    
    # Convert to numpy for visualization
    room_seg = torch.argmax(segmentation, dim=1).numpy()[0]
    icon_seg = torch.argmax(icon_pred, dim=1).numpy()[0]
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(np.moveaxis(fplan, 0, -1))
    
    plt.subplot(132)
    plt.title('Room Segmentation')
    segmentation_plot(room_seg)
    
    plt.subplot(133)
    plt.title('Icon Segmentation')
    segmentation_plot(icon_seg)
    
    # Save results
    plt.savefig(os.path.join(log_dir, 'prediction_results.png'))
    plt.close()
    
    print(f"Results saved to {log_dir}")
    
    # Get image dimensions from the resized input image
    height, width = fplan.shape[1], fplan.shape[2]  # since fplan is already channel-first
    img_size = (height, width)
    
    # Process predictions with post-processing
    heatmaps, rooms, icons = post_prosessing.split_prediction(pred, img_size, split)
    
    rooms_seg = np.argmax(rooms, axis=0)
    icons_seg = np.argmax(icons, axis=0)
    
    all_opening_types = [1, 2]  # Window, Door
    polygons, types, room_polygons, room_types = post_prosessing.get_polygons(
        (heatmaps, rooms, icons), 0.4, all_opening_types)
    
    # Get aspect ratio from input image
    input_aspect_ratio = fplan.shape[2] / fplan.shape[1]  # width/height
    
    # Calculate figure size maintaining aspect ratio
    fig_height = 15
    fig_width = fig_height * input_aspect_ratio
    
    # Create figure with correct aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.title('Polygon Results')
    plt.axis('off')
    
    # Plot room polygons
    colors = plt.cm.tab20(np.linspace(0, 1, len(room_cls)))  # Color map for rooms
    for polygon, room_type in zip(room_polygons, room_types):
        # Extract integer class index from room_type; it might be a dict.
        if isinstance(room_type, dict):
            class_idx = room_type.get('class', 0)
        else:
            class_idx = int(room_type)

        # If polygon is a Shapely polygon, get its exterior coordinates
        if hasattr(polygon, 'exterior'):
            polygon_coords = np.array(polygon.exterior.coords)
        else:
            polygon_coords = np.array(polygon)

        if len(polygon_coords) >= 3:  # Valid polygon must have at least 3 points
            plt.fill(polygon_coords[:, 0], polygon_coords[:, 1],
                    color=colors[class_idx],
                    alpha=0.5,
                    label=room_cls[class_idx])
            plt.plot(polygon_coords[:, 0], polygon_coords[:, 1],
                    color='black',
                    linewidth=1)
    
    # Plot opening polygons (windows, doors)
    for polygon, opening_type in zip(polygons, types):
        if hasattr(polygon, 'exterior'):
            polygon_coords = np.array(polygon.exterior.coords)
        else:
            polygon_coords = np.array(polygon)

        if len(polygon_coords) >= 3:
            plt.fill(polygon_coords[:, 0], polygon_coords[:, 1],
                    color='red' if opening_type == 1 else 'blue',  # red for windows, blue for doors
                    alpha=0.7)
            plt.plot(polygon_coords[:, 0], polygon_coords[:, 1],
                    color='black',
                    linewidth=1)
    
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='center left', 
              bbox_to_anchor=(1, 0.5))
    
    plt.savefig(os.path.join(log_dir, 'polygon_results.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    print(f"Polygon results saved to {log_dir}")
   