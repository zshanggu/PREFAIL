import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline

import time
from datetime import datetime

import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import cv2
import numpy as np
import sys

def flow_to_hsv(flow, max_magnitude=None):
    """
    Convert optical flow to HSV color representation.
    
    Args:
        flow: Optical flow array (H, W, 2) where flow[:,:,0] is x-displacement, flow[:,:,1] is y-displacement
        max_magnitude: Maximum magnitude for normalization. If None, uses the actual max in the flow.
    
    Returns:
        hsv_image: HSV image (H, W, 3) where hue represents direction and value represents magnitude
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Calculate magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Hue represents direction (0-180 in OpenCV)
    hsv[..., 0] = angle * 180 / np.pi / 2
    
    # Saturation is maxed out
    hsv[..., 1] = 255
    
    # Value represents magnitude (normalized)
    if max_magnitude is None:
        max_magnitude = np.max(magnitude)
    
    if max_magnitude > 0:
        hsv[..., 2] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)
    else:
        hsv[..., 2] = 0
    
    return hsv


def draw_dense_optical_flow(rgb_image, flow, max_magnitude=None, alpha=0.7):
    """
    Draw dense optical flow as an HSV color overlay on RGB image.
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        flow: Optical flow array (H, W, 2)
        max_magnitude: Maximum magnitude for normalization. If None, uses the actual max.
        alpha: Blending factor (0=original image, 1=full flow visualization)
    
    Returns:
        blended_image: RGB image with dense flow overlay
    """
    # Convert flow to HSV
    hsv = flow_to_hsv(flow, max_magnitude)
    
    # Convert HSV to RGB
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Blend with original image
    blended = cv2.addWeighted(rgb_image, 1-alpha, flow_rgb, alpha, 0)
    
    return blended


def create_flow_color_wheel(size=200):
    """
    Create a color wheel legend for optical flow visualization.
    
    Args:
        size: Size of the color wheel image
    
    Returns:
        RGB image showing the color wheel legend
    """
    # Create a coordinate grid
    y, x = np.mgrid[-size//2:size//2, -size//2:size//2]
    
    # Calculate angle and radius for each pixel
    angle = np.arctan2(y, x)
    radius = np.sqrt(x**2 + y**2)
    
    # Create HSV representation
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(255 * (1 - radius / (size//2)), 0, 255).astype(np.uint8)
    
    # Mask out pixels outside the circle
    mask = radius <= size // 2
    hsv[~mask] = 0
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Add directional labels
    center = size // 2
    radius_text = int(size * 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Right (red)
    cv2.putText(rgb, "R", (center + radius_text, center), font, font_scale, (255, 255, 255), font_thickness)
    # Up (yellow)
    cv2.putText(rgb, "U", (center, center - radius_text), font, font_scale, (255, 255, 255), font_thickness)
    # Left (cyan)
    cv2.putText(rgb, "L", (center - radius_text - 10, center), font, font_scale, (255, 255, 255), font_thickness)
    # Down (blue)
    cv2.putText(rgb, "D", (center, center + radius_text), font, font_scale, (255, 255, 255), font_thickness)
    
    return rgb


def draw_optical_flow(rgb_image, flow, step=16, scale=1.5, color=(0, 255, 0), thickness=2, min_magnitude=1.0):
    """
    Draw optical flow vectors on an RGB image (SPARSE visualization).
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        flow: Optical flow array (H, W, 2) where flow[:,:,0] is x-displacement, flow[:,:,1] is y-displacement
        step: Step size for sampling flow vectors (draw every step pixels)
        scale: Scale factor for arrow length
        color: BGR color tuple for the arrows
        thickness: Thickness of the arrow lines
        min_magnitude: Minimum flow magnitude to draw
    
    Returns:
        image_with_flow: RGB image with optical flow arrows drawn
    """
    image_with_flow = rgb_image.copy()
    h, w = rgb_image.shape[:2]
    
    # Create a grid of points to sample
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    # Get flow at these points
    fx, fy = flow[y, x].T
    
    # Create line endpoints
    lines = np.vstack([x, y, x + fx * scale, y + fy * scale]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    # Draw arrows
    for (x1, y1), (x2, y2) in lines:
        # Only draw if the flow magnitude is significant (avoid noise)
        magnitude = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if magnitude > min_magnitude:  # Threshold for minimum flow
            # Clamp coordinates to image bounds
            x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
            x2, y2 = max(0, min(x2, w-1)), max(0, min(y2, h-1))
            
            # Draw arrow
            cv2.arrowedLine(image_with_flow, (x1, y1), (x2, y2), 
                           color, thickness, tipLength=0.3)
    
    return image_with_flow


def calculate_optical_flow_visualization(prev_frame_rgb, curr_frame_rgb, 
                                         mode='sparse',
                                         flow_step=16, flow_scale=1.5,
                                         arrow_color=(0, 255, 0), min_magnitude=1.0,
                                         dense_alpha=0.7, dense_max_magnitude=None):
    """
    Calculate optical flow between two frames and create visualization.
    
    Args:
        prev_frame_rgb: Previous frame RGB image (H, W, 3)
        curr_frame_rgb: Current frame RGB image (H, W, 3)
        mode: Visualization mode - 'sparse', 'dense', or 'both'
        flow_step: Step size for sampling flow vectors (sparse mode)
        flow_scale: Scale factor for arrow length (sparse mode)
        arrow_color: BGR color for flow arrows (sparse mode)
        min_magnitude: Minimum flow magnitude to draw (sparse mode)
        dense_alpha: Blending factor for dense visualization (0=original, 1=full flow)
        dense_max_magnitude: Max magnitude for normalization (dense mode). If None, uses actual max.
    
    Returns:
        Visualization image with optical flow drawn on current frame
    """
    try:
        # Convert to grayscale for optical flow calculation
        prev_gray = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,      # Pyramid scale
            levels=3,            # Number of pyramid levels
            winsize=15,          # Averaging window size
            iterations=3,        # Number of iterations at each pyramid level
            poly_n=5,            # Size of pixel neighborhood
            poly_sigma=1.2,      # Standard deviation of Gaussian for derivative
            flags=0
        )
        
        # Create visualization based on mode
        if mode == 'sparse':
            # Draw sparse flow vectors (arrows)
            flow_vis = draw_optical_flow(curr_frame_rgb, flow, 
                                         step=flow_step, 
                                         scale=flow_scale,
                                         color=arrow_color,
                                         thickness=2, min_magnitude=min_magnitude)
        
        elif mode == 'dense':
            # Draw dense flow as HSV color overlay
            flow_vis = draw_dense_optical_flow(curr_frame_rgb, flow, 
                                              max_magnitude=dense_max_magnitude,
                                              alpha=dense_alpha)
        
        elif mode == 'both':
            # First draw dense flow
            flow_vis = draw_dense_optical_flow(curr_frame_rgb, flow, 
                                              max_magnitude=dense_max_magnitude,
                                              alpha=dense_alpha * 0.5)  # Reduce alpha to see arrows better
            # Then overlay sparse arrows
            flow_vis = draw_optical_flow(flow_vis, flow, 
                                        step=flow_step, 
                                        scale=flow_scale,
                                        color=arrow_color,
                                        thickness=2, min_magnitude=min_magnitude)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'sparse', 'dense', or 'both'")
        
        return flow_vis
        
    except Exception as e:
        print(f"Warning: Could not calculate optical flow: {e}")
        import traceback
        traceback.print_exc()
        return curr_frame_rgb.copy()


def add_optical_flow_to_episode_data(episodes_file_idx, data_idx,
                                     mode='sparse',
                                     flow_step=16, flow_scale=1.5, min_magnitude=1.0,
                                     dense_alpha=0.7):
    """
    Add optical flow visualizations to episode data at specific index.
    Calculates flow between previous frame and current frame.
    
    Args:
        episodes_file_idx: Episode data array
        data_idx: Current data index (must be > 0)
        mode: Visualization mode - 'sparse', 'dense', or 'both'
        flow_step: Step size for sampling flow vectors (sparse mode, default=16)
        flow_scale: Scale factor for arrow length (sparse mode, default=1.5)
        min_magnitude: Minimum flow magnitude to draw (default=1.0)
        dense_alpha: Blending factor for dense visualization (default=0.7)
    """
    if data_idx == 0:
        # No previous frame to compare against
        return
    
    try:
        # Get current and previous RGB images
        curr_top_rgb = episodes_file_idx[data_idx]['projected_top_frame_rgb']
        prev_top_rgb = episodes_file_idx[data_idx - 1]['projected_top_frame_rgb']
        
        curr_front_rgb = episodes_file_idx[data_idx]['projected_front_frame_rgb']
        prev_front_rgb = episodes_file_idx[data_idx - 1]['projected_front_frame_rgb']
        
        # Calculate and visualize optical flow for top camera
        top_flow_vis = calculate_optical_flow_visualization(
            prev_top_rgb, curr_top_rgb,
            mode=mode,
            flow_step=flow_step,
            flow_scale=flow_scale,
            arrow_color=(0, 255, 0),  # Green arrows
            min_magnitude=min_magnitude,
            dense_alpha=dense_alpha
        )
        
        # Calculate and visualize optical flow for front camera
        front_flow_vis = calculate_optical_flow_visualization(
            prev_front_rgb, curr_front_rgb,
            mode=mode,
            flow_step=flow_step,
            flow_scale=flow_scale,
            arrow_color=(0, 255, 0),  # Green arrows
            min_magnitude=min_magnitude,
            dense_alpha=dense_alpha
        )

        # cv2.imshow("Top camera optical flow", top_flow_vis)
        # cv2.imshow("Front camera optical flow", front_flow_vis)
        # cv2.waitKey(0) # Small delay to ensure the image is rendered

        # Store in episode data
        episodes_file_idx[data_idx]['top_optical_flow_viz'] = top_flow_vis
        episodes_file_idx[data_idx]['front_optical_flow_viz'] = front_flow_vis
        
    except KeyError as e:
        print(f"Warning: Missing RGB image key in frame {data_idx}: {e}")
        print(f"  Available keys: {list(episodes_file_idx[data_idx].keys())}")
    except Exception as e:
        print(f"Warning: Could not create optical flow visualization for frame {data_idx}: {e}")
        import traceback
        traceback.print_exc()

def draw_arrow_on_image(rgb_image, object_contour, panel_contour, 
                        relative_features, arrow_color=(0, 255, 0), 
                        arrow_thickness=3, scale_factor=1.0, camera_name="top"):
    """
    Draw an arrow representing the relative transformation on the RGB image.
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        object_contour: Object contour points (N, 2) or (N, 1, 2)
        panel_contour: Panel/gripper contour points (M, 2) or (M, 1, 2) - not used but kept for compatibility
        relative_features: [theta, dx, dy] transformation from object to panel
        arrow_color: BGR color tuple for the arrow (default: green)
        arrow_thickness: Thickness of the arrow line
        scale_factor: Scale factor for arrow length (increase if arrow is too small)
    
    Returns:
        image_with_arrow: RGB image with arrow drawn
    """
    # Make a copy to avoid modifying the original
    image_with_arrow = rgb_image.copy()
    
    # Helper function to check if contour is valid
    def is_valid_contour(contour):
        return (contour is not None and 
                len(contour) > 0 and 
                contour.shape[0] > 0 and 
                np.prod(contour.shape) > 0)
    
    if not is_valid_contour(object_contour):
        return image_with_arrow
    
    try:
        # Reshape contour if needed
        if len(object_contour.shape) == 3:  # (N, 1, 2) -> (N, 2)
            object_contour_reshaped = object_contour.reshape(-1, 2)
        else:
            object_contour_reshaped = object_contour
        
        # Calculate object centroid as starting point
        object_center = np.mean(object_contour_reshaped, axis=0)
        object_center = object_center.flatten()
        
        # Extract transformation values
        theta = relative_features[0]  # rotation angle in radians
        dx = relative_features[1]     # translation in x
        dy = relative_features[2]     # translation in y
        
        # Calculate arrow end point based on transformation
        # Start from object center, apply translation scaled for visibility
        start_x = int(object_center.item(0))
        start_y = int(object_center.item(1))
        end_x = int(start_x + dx * scale_factor)
        end_y = int(start_y + dy * scale_factor)
        
        # Validate bounds
        h, w = image_with_arrow.shape[:2]
        if (0 <= start_x < w and 0 <= start_y < h):
            
            # Clamp end point to image bounds
            end_x = max(0, min(end_x, w - 1))
            end_y = max(0, min(end_y, h - 1))
            
            # Draw translation arrow (main arrow)
            cv2.arrowedLine(image_with_arrow, 
                           (start_x, start_y), 
                           (end_x, end_y),
                           arrow_color, 
                           arrow_thickness, 
                           tipLength=0.3)
            
            # Draw rotation indicator (arc) - more prominent and clear
            if abs(theta) > 0.01:  # Only draw if there's significant rotation
                # Draw a larger, more visible arc to indicate rotation
                arc_radius = 60  # Large radius for visibility
                arc_thickness = 5  # Thick arc
                
                # Convert rotation angle to degrees for cv2.ellipse
                theta_deg = np.degrees(theta)
                
                # Draw a semi-transparent filled sector for large rotations
                if abs(theta_deg) > 15:  # If rotation is significant
                    overlay = image_with_arrow.copy()
                    
                    # Create points for the filled sector
                    num_points = max(int(abs(theta_deg)), 20)
                    angles = np.linspace(0 if theta > 0 else theta, theta if theta > 0 else 0, num_points)
                    sector_points = [(start_x, start_y)]
                    for angle in angles:
                        x = start_x + int(arc_radius * np.cos(angle))
                        y = start_y + int(arc_radius * np.sin(angle))
                        sector_points.append((x, y))
                    sector_points.append((start_x, start_y))
                    
                    # Draw filled sector
                    cv2.fillPoly(overlay, [np.array(sector_points, dtype=np.int32)], (0, 255, 255))
                    cv2.addWeighted(overlay, 0.3, image_with_arrow, 0.7, 0, image_with_arrow)
                
                # Draw multiple concentric arcs for better visibility
                for offset in [0, 8, 16]:  # Three concentric arcs with larger spacing
                    current_radius = arc_radius + offset
                    
                    if theta > 0:  # Counter-clockwise (positive rotation)
                        cv2.ellipse(image_with_arrow, 
                                   (start_x, start_y),
                                   (current_radius, current_radius),
                                   0, 0, theta_deg,
                                   (0, 255, 255),  # Cyan for rotation
                                   arc_thickness if offset == 0 else 3)
                    else:  # Clockwise (negative rotation)
                        cv2.ellipse(image_with_arrow, 
                                   (start_x, start_y),
                                   (current_radius, current_radius),
                                   0, theta_deg, 0,
                                   (0, 255, 255),  # Cyan for rotation
                                   arc_thickness if offset == 0 else 3)
                
                # Add arrow at the end of the arc to show rotation direction
                arc_end_angle = theta if theta > 0 else 0
                arc_end_x = start_x + int((arc_radius + 16) * np.cos(arc_end_angle))
                arc_end_y = start_y + int((arc_radius + 16) * np.sin(arc_end_angle))
                
                # Calculate arrow direction (tangent to the circle)
                if theta > 0:  # Counter-clockwise
                    arrow_dx = -np.sin(arc_end_angle) * 25
                    arrow_dy = np.cos(arc_end_angle) * 25
                else:  # Clockwise
                    arrow_dx = np.sin(arc_end_angle) * 25
                    arrow_dy = -np.cos(arc_end_angle) * 25
                
                arrow_end_x = arc_end_x + int(arrow_dx)
                arrow_end_y = arc_end_y + int(arrow_dy)
                
                # Draw the rotation direction arrow
                cv2.arrowedLine(image_with_arrow,
                               (arc_end_x, arc_end_y),
                               (arrow_end_x, arrow_end_y),
                               (0, 255, 255),  # Cyan
                               4,
                               tipLength=0.5)
                
            
            # Draw center point (larger and more visible)
            cv2.circle(image_with_arrow, (start_x, start_y), 7, (255, 0, 0), -1)
            cv2.circle(image_with_arrow, (start_x, start_y), 10, (255, 255, 255), 2)  # White outline
            
            
    except Exception as e:
        print(f"Warning: Could not draw arrow: {e}")
        import traceback
        traceback.print_exc()
    
    # cv2.imshow(f"{camera_name} camera view with arrow", image_with_arrow)
    # cv2.waitKey(0)  # Small delay to ensure the image is rendered
    # sys.exit(0)
    return image_with_arrow


def add_visualization_to_episode_data(episodes_file_idx, data_idx, 
                                       top_object_contour, top_panel_contour,
                                       front_object_contour, front_panel_contour,
                                       top_relative_to_init, front_relative_to_init,
                                       scale_factor=1.0):
    """
    Add visualization images to episode data at specific index.
    Draws arrows representing the relative transformation on the actual camera RGB images.
    
    Args:
        episodes_file_idx: Episode data array
        data_idx: Current data index
        top_object_contour, top_panel_contour: Top view contours
        front_object_contour, front_panel_contour: Front view contours
        top_relative_to_init: [theta, dx, dy] transformation for top view
        front_relative_to_init: [theta, dx, dy] transformation for front view
        scale_factor: Scale factor for arrow length (increase if arrows are too small, default=1.0)
    """
    try:
        # Get the original RGB images from the episode data
        top_rgb = episodes_file_idx[data_idx]['full_top_frame_rgb']
        front_rgb = episodes_file_idx[data_idx]['full_front_frame_rgb']
        
        # Draw arrows representing the transformations on the images
        top_vis = draw_arrow_on_image(
            top_rgb, 
            top_object_contour, 
            top_panel_contour,
            top_relative_to_init,
            arrow_color=(0, 255, 0),  # Green arrow for translation
            arrow_thickness=3,
            scale_factor=scale_factor,
            camera_name="top"
        )
        
        front_vis = draw_arrow_on_image(
            front_rgb,
            front_object_contour, 
            front_panel_contour,
            front_relative_to_init,
            arrow_color=(0, 255, 0),  # Green arrow for translation
            arrow_thickness=3,
            scale_factor=scale_factor,
            camera_name="front"
        )
        
        # Store in episode data
        episodes_file_idx[data_idx]['top_relative_to_init_viz'] = top_vis
        episodes_file_idx[data_idx]['front_relative_to_init_viz'] = front_vis
        
    except KeyError as e:
        print(f"Warning: Missing RGB image key in frame {data_idx}: {e}")
        print(f"  Available keys: {list(episodes_file_idx[data_idx].keys())}")
    except Exception as e:
        print(f"Warning: Could not create visualization for frame {data_idx}: {e}")

# Define different interpolation methods
def flat_interpolation(first_failure_time_step, failure_time_step_trim):
    # Calculate number of points needed
    num_points = int(first_failure_time_step - failure_time_step_trim + 1)
    # Return array of constant value 0.5 with the required length
    return np.full(num_points, 0.5)

def linear_interpolation(first_failure_time_step, failure_time_step_trim):
    return np.linspace(0, 1, first_failure_time_step - failure_time_step_trim + 1)

def sin_interpolation(first_failure_time_step, failure_time_step_trim):
    x = np.linspace(0, 1, first_failure_time_step - failure_time_step_trim + 1)
    return np.sin(x)

def spline_interpolation(start_idx, middle_idx, end_idx):
    """
    Create a cubic spline interpolation that:
    - Starts at (start_idx, 0.0)
    - Passes through (middle_idx, 0.5)
    - Ends at (end_idx, 1.0)
    - Stays within (0.0, 1.0)
    - Has continuous slope at the joint points
    
    Args:
        start_idx: The starting index (value will be 0.0)
        middle_idx: The index where value should be 0.5
        end_idx: The index where value should be 1.0
        
    Returns:
        Array of interpolated values from start_idx to end_idx
    """
    # Define the key points
    x = np.array([start_idx, middle_idx, end_idx])
    y = np.array([0.0, 0.5, 1.0])
    
    # Create the spline
    cs = CubicSpline(x, y, bc_type='natural')
    
    # Generate interpolated values
    x_interp = np.arange(start_idx, end_idx + 1)
    result = cs(x_interp)
    
    # Ensure values stay within [0.0, 1.0]
    result = np.clip(result, 0.0, 1.0)
    
    return result

def hermite_spline_interpolation(start_idx, middle_idx, end_idx):
    """
    Create a Hermite spline interpolation that:
    - Starts at (start_idx, 0.0) with zero slope
    - Passes through (middle_idx, 0.5) with calculated slope
    - Ends at (end_idx, 1.0) with zero slope
    - Guarantees values stay within [0.0, 1.0] without clipping
    - Has continuous first derivatives at the joint points
    """
    # Create output array
    num_points = end_idx - start_idx + 1
    result = np.zeros(num_points)
    
    # Define points and tangents for Hermite splines
    x = np.array([start_idx, middle_idx, end_idx])
    y = np.array([0.0, 0.5, 1.0])
    
    # Calculate reasonable tangents that will keep the curve bounded
    m1 = 1.0 / (middle_idx - start_idx) * 0.5  # Slope from point 0 to 1
    m2 = 1.0 / (end_idx - middle_idx) * 0.5   # Slope from point 1 to 2
    
    # Set tangents: nearly zero at ends, moderate at middle point
    tangents = np.array([0.1 * m1, (m1 + m2) / 2, 0.1 * m2])
    
    # Define Hermite basis functions
    def h00(t): return 2*t**3 - 3*t**2 + 1
    def h10(t): return t**3 - 2*t**2 + t
    def h01(t): return -2*t**3 + 3*t**2
    def h11(t): return t**3 - t**2
    
    # Loop through each segment
    for i in range(num_points):
        idx = start_idx + i
        
        if idx <= middle_idx:
            # First segment: 0.0 to 0.5
            t = (idx - start_idx) / (middle_idx - start_idx)
            
            # Hermite interpolation formula
            p0, p1 = y[0], y[1]
            m0, m1 = tangents[0] * (middle_idx - start_idx), tangents[1] * (middle_idx - start_idx)
            
            result[i] = h00(t) * p0 + h10(t) * m0 + h01(t) * p1 + h11(t) * m1
        else:
            # Second segment: 0.5 to 1.0
            t = (idx - middle_idx) / (end_idx - middle_idx)
            
            # Hermite interpolation formula
            p0, p1 = y[1], y[2]
            m0, m1 = tangents[1] * (end_idx - middle_idx), tangents[2] * (end_idx - middle_idx)
            
            result[i] = h00(t) * p0 + h10(t) * m0 + h01(t) * p1 + h11(t) * m1
    
    return result

def piecewise_bezier_interpolation(start_idx, middle_idx, end_idx):
    """
    Create a piecewise Bezier curve interpolation with edge case handling.
    
    Args:
        start_idx: The starting index (value will be 0.0)
        middle_idx: The index where value should be 0.5
        end_idx: The index where value should be 1.0
        
    Returns:
        Array of interpolated values from start_idx to end_idx
    """
    # Check edge cases
    if middle_idx == start_idx:
        # Handle case where middle point is at the start
        # In this case, we only need to interpolate from 0.5 to 1.0
        num_points = end_idx - start_idx + 1
        result = np.zeros(num_points)
        result[0] = 0.5  # Set the first point to 0.5
        
        # For points after the first one
        if num_points > 1:
            segment_length = end_idx - middle_idx
            
            for i in range(1, num_points):
                idx = start_idx + i
                # Calculate t - avoid division by zero
                t = (idx - middle_idx) / segment_length if segment_length > 0 else 1.0
                
                # Control points for Bezier curve
                q1y = 0.65  # Control point 1 y-value
                q2y = 0.85  # Control point 2 y-value
                
                # Cubic Bezier formula for second segment
                result[i] = (1-t)**3 * 0.5 + \
                           3*(1-t)**2*t * q1y + \
                           3*(1-t)*t**2 * q2y + \
                           t**3 * 1.0
        return result
        
    elif middle_idx == end_idx:
        # Handle case where middle point is at the end
        # In this case, we only need to interpolate from 0.0 to 0.5
        num_points = end_idx - start_idx + 1
        result = np.zeros(num_points)
        
        segment_length = middle_idx - start_idx
        
        for i in range(num_points - 1):
            idx = start_idx + i
            # Calculate t - this should be safe as we checked segment_length > 0
            t = (idx - start_idx) / segment_length
            
            # Control points for Bezier curve
            p1y = 0.15  # Control point 1 y-value
            p2y = 0.35  # Control point 2 y-value
            
            # Cubic Bezier formula for first segment
            result[i] = (1-t)**3 * 0.0 + \
                       3*(1-t)**2*t * p1y + \
                       3*(1-t)*t**2 * p2y + \
                       t**3 * 0.5
        
        # Set the last point to 0.5
        result[-1] = 0.5
        return result
    
    # Normal case - both segments exist
    num_points = end_idx - start_idx + 1
    result = np.zeros(num_points)
    
    # Calculate segment lengths
    segment1_length = middle_idx - start_idx
    segment2_length = end_idx - middle_idx
    
    # Calculate Bezier curve points
    for i in range(num_points):
        idx = start_idx + i
        
        if idx <= middle_idx:
            # First segment: 0.0 to 0.5
            t = (idx - start_idx) / segment1_length
            
            # Control points for Bezier curve
            p1y = 0.15  # Control point 1 y-value
            p2y = 0.35  # Control point 2 y-value
            
            # Cubic Bezier formula
            result[i] = (1-t)**3 * 0.0 + \
                       3*(1-t)**2*t * p1y + \
                       3*(1-t)*t**2 * p2y + \
                       t**3 * 0.5
        else:
            # Second segment: 0.5 to 1.0
            t = (idx - middle_idx) / segment2_length
            
            # Control points for Bezier curve
            q1y = 0.65  # Control point 1 y-value
            q2y = 0.85  # Control point 2 y-value
            
            # Cubic Bezier formula
            result[i] = (1-t)**3 * 0.5 + \
                       3*(1-t)**2*t * q1y + \
                       3*(1-t)*t**2 * q2y + \
                       t**3 * 1.0
    
    return result

def find_closest_points(src_points, dst_points):
    """
    Find closest point pairs between source and destination point sets.
    
    Args:
        src_points: (N, 2) array of source points
        dst_points: (M, 2) array of destination points
        
    Returns:
        Tuple of matched points arrays, both of shape (K, 2)
    """
    # Calculate pairwise distances between all points
    distances = cdist(src_points, dst_points)
    
    # Find closest destination point for each source point
    closest_indices = np.argmin(distances, axis=1)
    
    # Create matched pairs
    src_matched = src_points
    dst_matched = dst_points[closest_indices]
    
    return src_matched, dst_matched


def extract_points_from_mask(mask, show=False):
    """Extracts edge points from a binary mask."""
    
    mask = cv2.medianBlur(mask, 7)
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 100, L2gradient = True)
    if show:
        cv2.imshow(f"{datetime.datetime.now()}", edges)
    points = np.column_stack(np.where(edges > 0))

    # Swap columns to convert from (y, x) to (x, y)
    points = points[:, [1, 0]]
    # print(len(points))
    return points

def calculate_transformation(src_points, dst_points):
    """
    Calculate rigid transformation (R, t) between matched point sets.
    
    Args:
        src_points: (N, 2) array of source points
        dst_points: (N, 2) array of destination points
        
    Returns:
        R: 2x2 rotation matrix
        t: 2D translation vector
    """
    # Calculate centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    
    # Center the point sets
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # Calculate covariance matrix
    H = src_centered.T @ dst_centered
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation
    t = dst_centroid - (R @ src_centroid)
    
    return R, t

def create_homogeneous_matrix(R, t):
    """
    Create a 3x3 homogeneous transformation matrix from rotation R and translation t.
    
    Args:
        R: 2x2 rotation matrix
        t: 2D translation vector
        
    Returns:
        3x3 homogeneous transformation matrix
    """
    H = np.eye(3)
    H[:2, :2] = R
    H[:2, 2] = t
    return H

def icp_2d(src_contour, dst_contour, max_iterations=16, tolerance=1e-6, matrix=False):
    """
    Perform 2D ICP algorithm between two contours.
    
    Args:
        src_contour: Source contour points from OpenCV findContours
        dst_contour: Destination contour points from OpenCV findContours
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for mean squared error
        matrix: return R matrix if true, otherwise return rotation angle
        
    Returns:
        rotation_angle: Rotation angle in radians
        translation: 2D translation vector [tx, ty]
        error: Final mean squared error
    """
    # Convert contours to point arrays
    src_points = src_contour.reshape(-1, 2).astype(np.float32)
    dst_points = dst_contour.reshape(-1, 2).astype(np.float32)
    
    # Initialize transformation
    R_total = np.eye(2)
    t_total = np.zeros(2)
    
    prev_error = float('inf')
    current_error = float('inf')  # Initialize current_error
    
    # Check if contours are empty or too small
    if len(src_points) < 2 or len(dst_points) < 2:
        return R_total if matrix else np.arctan2(R_total[1, 0], R_total[0, 0]), t_total, current_error
    
    for iteration in range(max_iterations):
        # Find closest point pairs
        src_matched, dst_matched = find_closest_points(src_points, dst_points)
        
        # Calculate transformation
        R, t = calculate_transformation(src_matched, dst_matched)
        
        # Update total transformation
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Apply transformation to source points
        src_points = (R @ src_points.T).T + t
        
        # Calculate error
        current_error = np.mean(np.sum((src_matched - dst_matched) ** 2, axis=1))
        
        # Check convergence
        if abs(prev_error - current_error) < tolerance:
            break
        
        prev_error = current_error

    # print ("R_total[1, 0], R_total[0, 0]", R_total[1, 0], R_total[0, 0])

    if matrix:
        # Apply log1p scaling to R_total and t_total
        scale_factor = 1e5
        R_total_scaled = np.sign(R_total) * np.log1p(np.abs(R_total) * scale_factor)
        t_total_scaled = np.sign(t_total) * np.log1p(np.abs(t_total) * scale_factor)
        
        # print("Original R_total:", R_total)
        # print("Scaled R_total:", R_total_scaled)
        # print("Original t_total:", t_total)
        # print("Scaled t_total:", t_total_scaled)
        

        # print("H with scaled values:", H)
        
        return R_total_scaled, t_total_scaled, current_error
    else:
        # Extract rotation angle from rotation matrix
        # For a 2D rotation matrix R = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        # The rotation angle can be calculated as θ = atan2(R[1,0], R[0,0])
        rotation_angle = np.arctan2(R_total[1, 0], R_total[0, 0])  # radians
        
        # Apply the same log1p scaling if needed
        # scale_factor = 1e5
        # rotation_angle_scaled = np.sign(rotation_angle) * np.log1p(np.abs(rotation_angle) * scale_factor)
        # t_total_scaled = np.sign(t_total) * np.log1p(np.abs(t_total) * scale_factor)
        
        # print ("rotation_angle, t_total", rotation_angle, t_total)

        return rotation_angle, t_total, current_error

def process_consecutive_frames(contours1, contours2, matrix=False):
    """
    Process consecutive frames and calculate transformations for each contour pair.
    
    Args:
        contours1: List of contours from first frame
        contours2: List of contours from second frame
        matrix: return R matrix if true, otherwise return rotation angle
        
    Returns:
        List of (R, t, error) tuples for each matched contour pair
    """
    results = []
    # print("contours1 shape", contours1.shape)
    
    # Calculate ICP between matched contours
    r_total, t_total, error = icp_2d(contours1[0], contours2[0], matrix=matrix)
    if matrix:
        # Create homogeneous transformation matrix with scaled values
        H = create_homogeneous_matrix(r_total, t_total)
        results=[H, error]
    else:
        results=[r_total, t_total[0], t_total[1]]

    try:
        # Check for empty transforms
        if len(results) == 0:
            raise Exception("All transforms are empty")
    except Exception as e:
        print(f"Error in contour processing: {e}")
    
    if matrix:
        features = extract_transform_features(results)
    else:
        features = results

    return features

def extract_transform_features(transforms):
    if not transforms:
        # Return zeros if no transforms
        return np.zeros(6)  # 4 for rotation matrix elements + 2 for translation
    
    # Take first transform if multiple are present
    H, error = transforms
    
    # Extract rotation matrix and translation vector
    R = H[:2, :2]  # 2x2 rotation matrix
    t = H[:2, 2]   # 2D translation vector
    
    # Convert to numpy arrays if needed
    R = np.array(R)
    t = np.array(t)
    
    # Ensure correct shapes before concatenating
    R_flat = R.flatten()  # Make 2x2 matrix into 1D array of length 4
    t = t.reshape(-1)    # Ensure translation is 1D array
    
    # print("R_flat", R_flat)
    # print("t", t)
    # Combine into single feature vector
    features = np.concatenate([R_flat, t])
    features = [features[0], features[1], features[4], features[5]]
    # print("features", features)
    
    return features

def process_real_camera_mask(masks, min_contour_area=100):
    """
    Process camera frame to get mask and contours for non-zero pixel values.
    
    Args:
        frame: RGB image array (height, width, 3)
        min_contour_area: 100  # Adjust this threshold as needed
    
    Returns:
        tuple: (mask, contours, filtered_image)
    """
    
    # Find contours
    contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours (noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    
    # If we have valid contours, keep only the largest one
    if filtered_contours:
        # Find the contour with the largest area
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        # Use only this contour
        filtered_contours = [largest_contour]
        
    else:
        # If no valid contours found, create a default contour (small square)
        # default_contour = np.array([[[10, 10]], [[10, 20]], [[20, 20]], [[20, 10]]], dtype=np.int32)
        # filtered_contours = [default_contour]
        # cv2.drawContours(filtered_image, filtered_contours, -1, (0, 0, 255), 2)  # Red color for default
        # print("No valid contours found")
        pass
    
    return filtered_contours

def process_simulation_camera_frame(frame, min_contour_area=100):
    """
    Process camera frame to get mask and contours for non-zero pixel values.
    
    Args:
        frame: RGB image array (height, width, 3)
        min_contour_area: 100  # Adjust this threshold as needed
    
    Returns:
        tuple: (mask, contours, filtered_image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Create mask for non-zero pixels
    # Use a small threshold to filter out near-black pixels
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours (noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Create visualization of the filtered image
    filtered_image = frame.copy()
    
    # If we have valid contours, keep only the largest one
    if filtered_contours:
        # Find the contour with the largest area
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        # Use only this contour
        filtered_contours = [largest_contour]
        
        # Draw the contour on the filtered image
        cv2.drawContours(filtered_image, filtered_contours, -1, (0, 255, 0), 2)
    else:
        # If no valid contours found, create a default contour (small square)
        # default_contour = np.array([[[10, 10]], [[10, 20]], [[20, 20]], [[20, 10]]], dtype=np.int32)
        # filtered_contours = [default_contour]
        # cv2.drawContours(filtered_image, filtered_contours, -1, (0, 0, 255), 2)  # Red color for default
        # print("No valid contours found")
        pass
    
    return mask, filtered_contours, filtered_image

# Function to calculate failure phase based on displacement, linear speed, and angular speed
def calculate_failure_phase(displacement, object_pos, panel_pos):
    """
    Calculate failure phase based on object position relative to panel.
    
    Args:
        displacement: Current displacement vector between object and panel
        object_pos: 3D position of the object (bunny)
        panel_pos: 3D position of the panel
        
    Returns:
        float: failure phase (0.0 = safe on panel, 1.0 = fallen off panel)
    """
    # print("displacement", displacement)
    # Check if object is below the panel surface (has fallen)
    if object_pos[2] < panel_pos[2] - 0.02:  # Small threshold to account for contact
        return 1.0
    
    # Check horizontal displacement (if object center is too far from panel center)
    # panel_radius = 0.15  # Approximate radius of panel - adjust based on your model
    # horizontal_dist = np.sqrt(displacement[0]**2 + displacement[1]**2)
    
    # if horizontal_dist > panel_radius:
    #     return 1.0  # Object center is outside panel radius
        
    # Object is safely on the panel
    return 0.0

def resample_data(episode, cut=True, scale=2):
    episode_resampled = []
    for item_idx in range(len(episode)):
        # print(episode[item_idx]['failure_phase_value'][0])
        if episode[item_idx]['failure_phase_value'][0] == 0.0 and item_idx%scale==0:
            episode_resampled.append(episode[item_idx])
        elif episode[item_idx]['failure_phase_value'][0] > 0.0 and episode[item_idx]['failure_phase_value'][0] < 1.0:
            episode_resampled.append(episode[item_idx])
        elif episode[item_idx]['failure_phase_value'][0] == 1.0 and item_idx%scale==0 and not cut:
            episode_resampled.append(episode[item_idx])
        else:
            pass
    return episode_resampled
                    
def plot_raw_metrics(original_episode, episode_num, dataset_type, save_path, current_object_name, timestamp):
    """
    Visualize original and resampled failure phase curves in a single plot.
    
    Parameters:
    -----------
    original_episode : list
        Original episode data
    resampled_episode : list
        Resampled episode data
    episode_num : int
        Episode number for the filename
    """
    # Extract original data
    original_time_steps = range(len(original_episode))
    original_risk_values = [item['failure_phase_value'][0] for item in original_episode]
    
    # Create figure and plot
    plt.figure(figsize=(12, 6))
    
    # Plot original data as a continuous line
    plt.plot(original_time_steps, original_risk_values, 'b-', 
            linewidth=2, alpha=0.6, label='Original data')
    
    # Enhance the plot
    plt.title('Original failure phases')
    plt.xlabel('Time Step')
    plt.ylabel('Risk Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/episode{episode_num}_{current_object_name}_{dataset_type}_{timestamp}_raw.png')
    plt.close()

def plot_metrics(original_episode, resampled_episode, save_path, episode_name=None):
    """
    Visualize original and resampled failure phase curves in a single plot.
    
    Parameters:
    -----------
    original_episode : list
        Original episode data
    resampled_episode : list
        Resampled episode data
    episode_num : int
        Episode number for the filename
    """
    # Extract original data
    # original_time_steps = range(len(original_episode))
    # original_risk_values = [item['risk'][0] for item in original_episode]
    
    # Map resampled points to their original indices
    resampled_indices = []
    for r_item in resampled_episode:
        # Find matching item in original episode
        for i, o_item in enumerate(original_episode):
            if np.array_equal(r_item['time_step'], o_item['time_step']) and r_item['risk'][0] == o_item['risk'][0]:
                resampled_indices.append(i)
                break
    
    # Extract resampled values
    resampled_risk_values = [item['risk'][0] for item in resampled_episode]
    
    plt.rcParams.update({
            'font.size': 25,          # Base font size
            'axes.titlesize': 25,     # Title font size
            'axes.labelsize': 25,     # X and Y label font size
            'xtick.labelsize': 25,    # X tick label font size
            'ytick.labelsize': 25,    # Y tick label font size
            'legend.fontsize': 25,    # Legend font size
            'figure.titlesize': 25    # Figure title font size
        })
        
    # Create figure and plot
    plt.figure(figsize=(12, 6))
    
    # Plot original data as a continuous line
    # plt.plot(original_time_steps, original_risk_values, 'b-', 
    #         linewidth=2, alpha=0.6, label='Original data')
    
    # Plot resampled data as a dotted line with markers
    plt.plot(resampled_indices, resampled_risk_values, 'r--', 
            linewidth=1.5, alpha=0.8, label='Resampled curve')
    
    # Add markers for resampled points
    plt.scatter(resampled_indices, resampled_risk_values, 
                color='red', s=40, label='Resampled points', zorder=5)
    
    # Enhance the plot
    # plt.title('Comparison of Original and Resampled failure phases')
    plt.xlabel('Time Step')
    plt.ylabel('Risk Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text annotation showing reduction
    reduction = 100 * (1 - len(resampled_episode)/len(original_episode))
    plt.annotate(f"Data reduction: {reduction:.1f}%\nOriginal: {len(original_episode)} points\nResampled: {len(resampled_episode)} points", 
                xy=(0.98, 0.02), xycoords='axes fraction',  # Bottom-right instead of top-left
                ha='right', va='bottom',  # Proper text alignment
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
                fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/new/comparison_{episode_name}.png')
    plt.close()

def read_config(file_path):
    """
    Read and parse a JSON configuration file.
    
    Args:
        file_path (str): Path to the JSON config file
        
    Returns:
        dict: The parsed configuration as a dictionary
    """
    try:
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' contains invalid JSON.")
        return None
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None
    
def combine_arrays(arrays, start_idx=0, end_idx=None, operation='sum'):
    """
    Combine values in a series of NumPy arrays position by position.
    
    Parameters:
    -----------
    arrays : list of numpy.ndarray or list
        List of NumPy arrays (or lists) to combine
    start_idx : int, optional
        Starting index (inclusive) of arrays to combine, default is 0
    end_idx : int, optional
        Ending index (inclusive) of arrays to combine, default is None (which means all arrays)
    operation : str, optional
        Operation to perform: 'sum', 'mean', 'max', 'min', 'prod', default is 'sum'
    
    Returns:
    --------
    numpy.ndarray
        Combined array with the result of the operation
    """
    # Validate input
    if not arrays:
        raise ValueError("Input list is empty")
    
    # Set default end_idx if not provided
    if end_idx is None:
        end_idx = len(arrays) - 1  # Inclusive of the last element
    
    # Validate indices
    if start_idx < 0 or start_idx >= len(arrays):
        raise ValueError(f"start_idx {start_idx} is out of bounds for array of length {len(arrays)}")
    if end_idx < 0 or end_idx >= len(arrays):
        raise ValueError(f"end_idx {end_idx} is out of bounds for array of length {len(arrays)}")
    if start_idx > end_idx:
        raise ValueError(f"start_idx {start_idx} must be less than or equal to end_idx {end_idx}")
    
    # Select subset of arrays - note the +1 to make end_idx inclusive
    selected_arrays = arrays[start_idx:end_idx+1]
    
    # Convert all elements to numpy arrays if they aren't already
    selected_arrays = [np.array(arr) for arr in selected_arrays]
    
    # Check if all arrays have the same shape
    shape = selected_arrays[0].shape
    for i, arr in enumerate(selected_arrays[1:], 1):
        if arr.shape != shape:
            raise ValueError(f"Array at index {i+start_idx} has shape {arr.shape}, different from {shape}")
    
    # Stack arrays along a new axis
    stacked = np.stack(selected_arrays)
    
    # Perform the requested operation along the first axis (the stacking axis)
    if operation == 'sum':
        return np.sum(stacked, axis=0)
    elif operation == 'mean':
        return np.mean(stacked, axis=0)
    elif operation == 'max':
        return np.max(stacked, axis=0)
    elif operation == 'min':
        return np.min(stacked, axis=0)
    elif operation == 'prod':
        return np.prod(stacked, axis=0)
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def reshape_contours(contours):
    """
    Reshape contours to make them compatible with OpenCV functions.
    Handles different input shapes including nested contours.
    
    Args:
        contours: Contours in shape (1, N, 1, 2) or similar
        
    Returns:
        Reshaped contours as a list of OpenCV-compatible contours
    """
    reshaped_contours = []
    
    # Handle case where we have a single contour in shape (1, N, 1, 2)
    if isinstance(contours, np.ndarray) and contours.ndim == 4:
        for i in range(contours.shape[0]):
            # Extract each contour and reshape to (N, 1, 2)
            contour = contours[i].reshape(-1, 1, 2).astype(np.int32)
            reshaped_contours.append(contour)
    # Case where we already have a list of contours
    elif isinstance(contours, list):
        for contour in contours:
            if isinstance(contour, np.ndarray):
                # Make sure contour is in shape (N, 1, 2)
                if contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2:
                    reshaped_contours.append(contour.astype(np.int32))
                elif contour.ndim == 2 and contour.shape[1] == 2:
                    # Reshape (N, 2) to (N, 1, 2)
                    reshaped_contours.append(contour.reshape(-1, 1, 2).astype(np.int32))
                else:
                    raise ValueError(f"Unsupported contour shape: {contour.shape}")
    else:
        raise ValueError(f"Unsupported contours type: {type(contours)}")
    
    return reshaped_contours

def apply_transformation(contour, rotation, tx, ty):
    """
    Apply rotation and translation to a contour.
    
    Args:
        contour: Contour points array from OpenCV in shape (N, 1, 2)
        rotation: Rotation angle in radians
        tx, ty: Translation parameters
    
    Returns:
        Transformed contour in the same shape as input
    """
    # Extract points as (N, 2)
    points = contour.reshape(-1, 2).astype(np.float32)
    
    # Create rotation matrix
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    
    # Apply rotation and translation
    transformed_points = (R @ points.T).T + np.array([tx, ty])
    
    # Convert back to original contour format
    transformed_contour = transformed_points.reshape(contour.shape).astype(np.int32)
    
    return transformed_contour

def generate_sample_contours(shape1=(1, 56, 1, 2), shape2=(1, 53, 1, 2)):
    """Generate two sample contours with specified shapes for testing."""
    
    # Create a simple polygon for the first contour
    angles1 = np.linspace(0, 2*np.pi, shape1[1], endpoint=False)
    radius1 = 100 + 10 * np.sin(3 * angles1)  # Add some variation
    x1 = 200 + radius1 * np.cos(angles1)
    y1 = 200 + radius1 * np.sin(angles1)
    contour1_points = np.column_stack((x1, y1))
    
    # Reshape to match the requested shape
    contour1 = contour1_points.reshape(shape1).astype(np.int32)
    
    # Create a slightly different polygon for the second contour
    angles2 = np.linspace(0, 2*np.pi, shape2[1], endpoint=False)
    radius2 = 90 + 15 * np.sin(4 * angles2)  # Different variation
    x2 = 250 + radius2 * np.cos(angles2)  # Shifted center
    y2 = 220 + radius2 * np.sin(angles2)
    contour2_points = np.column_stack((x2, y2))
    
    # Reshape to match the requested shape
    contour2 = contour2_points.reshape(shape2).astype(np.int32)
    
    return contour1, contour2

def load_image_contours(image1_path, image2_path, target_shape1=(1, 56, 1, 2), target_shape2=(1, 53, 1, 2)):
    """
    Load contours from two images and reshape to target shapes.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        target_shape1: Target shape for first contour
        target_shape2: Target shape for second contour
        
    Returns:
        Two contours with specified shapes
    """
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Failed to load images: {image1_path} or {image2_path}")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    raw_contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour from each image
    if len(raw_contours1) > 0 and len(raw_contours2) > 0:
        # Select largest contours
        contour1 = max(raw_contours1, key=cv2.contourArea)
        contour2 = max(raw_contours2, key=cv2.contourArea)
        
        # Resample contours to match target shapes
        contour1 = resample_contour(contour1, target_shape1[1])
        contour2 = resample_contour(contour2, target_shape2[1])
        
        # Reshape to match target shapes
        contour1 = contour1.reshape(target_shape1).astype(np.int32)
        contour2 = contour2.reshape(target_shape2).astype(np.int32)
        
        return contour1, contour2
    else:
        raise ValueError("No contours found in one or both images")

def resample_contour(contour, target_points):
    """
    Resample a contour to have exactly the specified number of points.
    
    Args:
        contour: OpenCV contour
        target_points: Desired number of points
        
    Returns:
        Resampled contour with target_points points
    """
    # Convert to (N, 2) array
    points = contour.reshape(-1, 2)
    
    # Calculate the perimeter
    perimeter = cv2.arcLength(contour, True)
    
    # Create a new array for resampled points
    resampled = np.zeros((target_points, 2), dtype=np.int32)
    
    # Distance between each new point
    step = perimeter / target_points
    
    # Initialize
    dist_traveled = 0
    new_idx = 0
    resampled[0] = points[0]
    
    # Loop through original points
    for i in range(1, len(points)):
        # Distance to next original point
        dist = np.linalg.norm(points[i] - points[i-1])
        
        while dist_traveled + dist >= step and new_idx < target_points - 1:
            # Interpolate to find next point
            alpha = (step - dist_traveled) / (dist + 1e-10)
            new_idx += 1
            resampled[new_idx] = points[i-1] + alpha * (points[i] - points[i-1])
            dist -= step - dist_traveled
            dist_traveled = 0
        
        dist_traveled += dist
    
    # Ensure we have the exact number of points (handle any floating-point issues)
    if new_idx < target_points - 1:
        resampled[new_idx+1:] = points[-1]
    
    return resampled.reshape(-1, 1, 2)

def visualize_contour_transformation(processed_c1, processed_c2, features, data_idx, output_path="./", episode_name="episode"):
    """
    Visualize original contours and the transformation using pre-processed contours and features.
    
    Args:
        processed_c1: First contour, already processed for use with process_consecutive_frames
        processed_c2: Second contour, already processed for use with process_consecutive_frames
        features: Transformation features from process_consecutive_frames (rotation, tx, ty)
        output_path: Path to save the visualization
    """
    # Extract rotation and translation parameters
    rotation, tx, ty = features[0], features[1], features[2]
    # print(f"Transformation parameters - Rotation: {rotation:.4f} rad ({rotation * 180 / np.pi:.2f}°), Translation: ({tx:.2f}, {ty:.2f})")
    
    # Reshape contours for visualization if needed
    reshaped_c1 = processed_c1
    reshaped_c2 = processed_c2
    
    # Ensure contours are in the right format for visualization
    if isinstance(processed_c1, np.ndarray) and processed_c1.ndim > 3:
        reshaped_c1 = reshape_contours(processed_c1)
    elif isinstance(processed_c1, list):
        reshaped_c1 = processed_c1
    
    if isinstance(processed_c2, np.ndarray) and processed_c2.ndim > 3:
        reshaped_c2 = reshape_contours(processed_c2)
    elif isinstance(processed_c2, list):
        reshaped_c2 = processed_c2
        
    # Apply transformation to the first contour
    transformed_contours = []
    if isinstance(reshaped_c1, list):
        for c in reshaped_c1:
            transformed_contour = apply_transformation(c, rotation, tx, ty)
            transformed_contours.append(transformed_contour)
    else:
        # Handle case where reshaped_c1 is a single contour
        transformed_contours.append(apply_transformation(reshaped_c1, rotation, tx, ty))
    
    # Create canvas for visualization
    canvas_size = (800, 600)
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    
    # Ensure all contours are in the right format for cv2.drawContours
    draw_c1 = reshaped_c1 if isinstance(reshaped_c1, list) else [reshaped_c1]
    draw_c2 = reshaped_c2 if isinstance(reshaped_c2, list) else [reshaped_c2]
    
    # Draw contours
    cv2.drawContours(canvas, draw_c1, -1, (255, 0, 0), 2)  # Original contour1 in blue
    cv2.drawContours(canvas, draw_c2, -1, (0, 255, 0), 2)  # Original contour2 in green
    cv2.drawContours(canvas, transformed_contours, -1, (0, 0, 255), 2)  # Transformed contour1 in red
    
    # Add legend
    legend_y = 30
    cv2.putText(canvas, "Original Contour 1", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(canvas, "Original Contour 2", (20, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "Transformed Contour 1", (20, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add transformation parameters
    rotation_deg = rotation * 180 / np.pi
    cv2.putText(canvas, f"Rotation: {rotation_deg:.2f} degrees", (20, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Translation: ({tx:.2f}, {ty:.2f})", (20, legend_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save the visualization
    cv2.imwrite(f'{output_path}/new/transform_{episode_name}_{data_idx}.png', canvas)
    print(f"Visualization saved as {output_path}")
    
    # # Create figure and display/save the result
    # plt.figure(figsize=(10, 8))
    # plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title("Contour Transformation Visualization")
    # plt.axis('off')
    # plt.tight_layout()
    
    # # Save as PNG as well
    # output_png = os.path.splitext(output_path)[0] + "_plt.png"
    # plt.savefig(output_png)
    # plt.close()
