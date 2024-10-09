import cv2
import numpy as np
import os
# from scipy import ndimage
!pip install numpy
!pip install cv2
def compute_likelihood_map(frame, object_hypothesis, object_model):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    object_hist = object_model['histogram']
    likelihood_map = cv2.calcBackProject([hsv], [0, 1], object_hist, [0, 180, 0, 256], 1)
    likelihood_map = cv2.normalize(likelihood_map, None, 0, 1, cv2.NORM_MINMAX)
    return likelihood_map

def compute_cumulative_histograms(likelihood_map, object_region, surrounding_region):
    h_o = np.histogram(likelihood_map[tuple(object_region)], bins=256, range=(0, 1))[0]
    h_s = np.histogram(likelihood_map[tuple(surrounding_region)], bins=256, range=(0, 1))[0]
    c_o = np.cumsum(h_o) / np.sum(h_o)
    c_s = np.cumsum(h_s) / np.sum(h_s)
    return c_o, c_s

def adaptive_threshold(c_o, c_s):
    valid_thresholds = np.where(c_o + c_s >= 1)[0]
    costs = 2 * c_o[valid_thresholds] - np.append(c_o[valid_thresholds[1:]], [1]) + c_s[valid_thresholds]
    tau_star = valid_thresholds[np.argmin(costs)] / 255.0
    return tau_star

def estimate_scale(likelihood_map, object_hypothesis, tau):
    segmentation = likelihood_map > tau
    safe_region = np.zeros_like(segmentation)
    top, left, bottom, right = object_hypothesis
    safe_region[top:bottom, left:right] = True
    safe_region[top+1:bottom-1, left+1:right-1] = False  # Inner 80% (approximated)
    
    labeled, num_features = ndimage.label(segmentation)
    object_mask = np.zeros_like(segmentation)
    
    for label in range(1, num_features + 1):
        component = labeled == label
        if np.any(component & safe_region):
            avg_likelihood = np.mean(likelihood_map[component])
            if avg_likelihood > tau:
                object_mask |= component
    
    if np.sum(object_mask) == 0:
        return object_hypothesis
    
    rows, cols = np.where(object_mask)
    top, bottom = np.min(rows), np.max(rows)
    left, right = np.min(cols), np.max(cols)
    return np.array([top, left, bottom, right])

def update_object_hypothesis(prev_hypothesis, scale_estimate, lambda_s):
    return (lambda_s * scale_estimate + (1 - lambda_s) * prev_hypothesis).astype(int)

def update_object_model(frame, object_hypothesis):
    top, left, bottom, right = object_hypothesis
    object_region = frame[top:bottom, left:right]
    hsv = cv2.cvtColor(object_region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return {'histogram': hist}

def track_object(frames, initial_hypothesis, lambda_s=0.5, max_scale_change=0.2):
    object_hypothesis = initial_hypothesis
    object_model = None
    
    for frame in frames:
        if object_model is None:
            object_model = update_object_model(frame, object_hypothesis)
        
        likelihood_map = compute_likelihood_map(frame, object_hypothesis, object_model)
        
        top, left, bottom, right = object_hypothesis
        object_region = (slice(top, bottom), slice(left, right))
        surrounding_region = (slice(max(0, top-20), min(frame.shape[0], bottom+20)),
                              slice(max(0, left-20), min(frame.shape[1], right+20)))
        
        c_o, c_s = compute_cumulative_histograms(likelihood_map, object_region, surrounding_region)
        tau_star = adaptive_threshold(c_o, c_s)
        
        scale_estimate = estimate_scale(likelihood_map, object_hypothesis, tau_star)
        
        prev_area = np.prod(object_hypothesis[2:] - object_hypothesis[:2])
        new_area = np.prod(scale_estimate[2:] - scale_estimate[:2])
        print(prev_area, new_area)
        scale_change = np.abs(new_area / prev_area - 1)
        
        if scale_change <= max_scale_change:
            object_hypothesis = update_object_hypothesis(object_hypothesis, scale_estimate, lambda_s)
            object_model = update_object_model(frame, object_hypothesis)
        
        yield frame, object_hypothesis, likelihood_map

def main():
    sequence_folder = 'sequence/'
    frame_files = sorted([f for f in os.listdir(sequence_folder) if f.endswith('.jpg') or f.endswith('.png')])
    
    # Given object location
    initial_bbox = (246, 208, 23, 23)  # Converted to integer pixel values for bounding box
    
    initial_hypothesis = np.array([initial_bbox[1], initial_bbox[0], 
                                   initial_bbox[1] + initial_bbox[3], 
                                   initial_bbox[0] + initial_bbox[2]])
    
    frames = (cv2.imread(os.path.join(sequence_folder, f)) for f in frame_files)
    # for frame in frames:
    #     cv2.imshow('Object Tracking', frame)
    #     cv2.waitKey(0)
    
    for frame, object_hypothesis, likelihood_map in track_object(frames, initial_hypothesis):
        vis_frame = frame.copy()
        top, left, bottom, right = object_hypothesis
        cv2.rectangle(vis_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        color_map = cv2.applyColorMap((likelihood_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis_likelihood = cv2.addWeighted(frame, 0.5, color_map, 0.5, 0)
        
        vis = np.hstack((vis_frame, vis_likelihood))
        cv2.imshow('Object Tracking', vis)
        cv2.waitKey(0)
        # key = cv2.waitKey(30) & 0xFF
        # if key == 27:  # ESC key
        #     break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
