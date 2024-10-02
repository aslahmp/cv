import cv2
import numpy as np
import math
from scipy.signal import hann
from collections import deque
class DAT_TRACKER:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scale_factor_ = 1.0
        self.prob_lut_ =  np.random.rand(256, 256, 256).astype(np.float32) 
        self.prob_lut_distractor_ = None
        self.prob_lut_masked_ = None
        self.adaptive_threshold_ = None
        self.target_pos_history_ = []
        self.target_sz_history_ = []

    def pos2rect(self, target_pos, target_sz, img_size):
        """Convert target position and size to rectangle."""
        x = int(target_pos[0] - target_sz[0] / 2)
        y = int(target_pos[1] - target_sz[1] / 2)
        return (x, y, int(target_sz[0]), int(target_sz[1]))

    def get_subwindow(self, img, target_pos, surr_sz):
        """Get subwindow around target position."""
        x1 = int(target_pos[0] - surr_sz[0] / 2)
        y1 = int(target_pos[1] - surr_sz[1] / 2)
        x2 = int(target_pos[0] + surr_sz[0] / 2)
        y2 = int(target_pos[1] + surr_sz[1] / 2)

        # Ensure the coordinates are within the image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        return img[y1:y2, x1:x2]

    def get_foreground_background_probs(self, surr_win, obj_rect_surr, num_bins, bin_mapping):
        """Placeholder function for getting foreground/background probabilities."""
        # This function needs to be implemented based on your tracking algorithm
        # For now, let's return a dummy probability map
        prob_map = np.zeros(surr_win.shape[:2], dtype=np.float32)
        return prob_map

    def get_adaptive_threshold(self, prob_map, obj_rect_surr):
        """Placeholder function for getting adaptive threshold."""
        # This function needs to be implemented based on your tracking algorithm
        # For now, let's return a dummy adaptive threshold
        return np.mean(prob_map)

    def tracker_dat_initialize(self, I, region):
        cx = region[0] + (region[2] - 1) / 2.0
        cy = region[1] + (region[3] - 1) / 2.0
        w = region[2]
        h = region[3]

        target_pos = (round(cx), round(cy))
        target_sz = (round(w), round(h))

        # Scale factor calculation
        diag = math.sqrt(target_sz[0] ** 2 + target_sz[1] ** 2)
        self.scale_factor_ = min(1.0, round(10.0 * self.cfg['img_scale_target_diagonal'] / diag) / 10.0)
        
        target_pos = (int(target_pos[0] * self.scale_factor_), int(target_pos[1] * self.scale_factor_))
        target_sz = (int(target_sz[0] * self.scale_factor_), int(target_sz[1] * self.scale_factor_))

        # Resize the input image
        img = cv2.resize(I, None, fx=self.scale_factor_, fy=self.scale_factor_)

        # Color space conversion based on configuration
        if self.cfg['color_space'] == 1:  # RGB
            img = img.copy()
        elif self.cfg['color_space'] == 2:  # LAB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        elif self.cfg['color_space'] == 3:  # HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.cfg['color_space'] == 4:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print("color_space does not equal any of the above cases")

        # Create the surrounding window size
        surr_sz = (int(self.cfg['surr_win_factor'] * target_sz[0]),
                   int(self.cfg['surr_win_factor'] * target_sz[1]))

        surr_rect = self.pos2rect(target_pos, surr_sz, img.shape[:2])
        obj_rect_surr = self.pos2rect(target_pos, target_sz, img.shape[:2])

        # Adjust object rectangle based on the surrounding rectangle
        obj_rect_surr = (obj_rect_surr[0] - surr_rect[0], obj_rect_surr[1] - surr_rect[1],
                         obj_rect_surr[2], obj_rect_surr[3])

        # Get the subwindow of the image for tracking
        surr_win = self.get_subwindow(img, target_pos, surr_sz)

        # Get foreground and background probabilities
        prob_map = self.get_foreground_background_probs(surr_win, obj_rect_surr, 
                                                        self.cfg['num_bins'], self.cfg['bin_mapping'])

        # Clone the probability look-up tables (placeholders)
        self.prob_lut_distractor_ = self.prob_lut_.copy() if self.prob_lut_ is not None else None
        self.prob_lut_masked_ = self.prob_lut_.copy() if self.prob_lut_ is not None else None

        # Compute adaptive threshold based on the probability map
        self.adaptive_threshold_ = self.get_adaptive_threshold(prob_map, obj_rect_surr)

        # Save the target position and size history for tracking
        self.target_pos_history_.append((target_pos[0] / self.scale_factor_, target_pos[1] / self.scale_factor_))
        self.target_sz_history_.append((target_sz[0] / self.scale_factor_, target_sz[1] / self.scale_factor_))
    def get_foreground_prob(self, frame, prob_lut, bin_mapping):
        # Ensure the input image is 3-channel (RGB)
        if frame.shape[2] != 3:
            raise ValueError("Expected a 3-channel (RGB) image")

        # Ensure bin_mapping is 1D with 256 elements
        print(bin_mapping.shape)
        if bin_mapping.size != 256 or bin_mapping.ndim != 1:
            raise ValueError("bin_mapping must be a 1D array with 256 elements")

        # Apply bin mapping to each channel using cv2.LUT
        frame_bin = np.zeros_like(frame)
        for i in range(3):  # Iterate over each channel (B, G, R)
            frame_bin[:, :, i] = cv2.LUT(frame[:, :, i], bin_mapping)

        # Initialize the probability map
        prob_map = np.zeros(frame.shape[:2], dtype=np.float32)

        # Iterate over each pixel in the frame_bin
        height, width = frame.shape[:2]
        for y in range(height):
            for x in range(width):
                # Get the bin-mapped values for each pixel
                b, g, r = frame_bin[y, x]
                # Use the bin-mapped values to index into the prob_lut and fill the prob_map
                prob_map[y, x] = prob_lut[b, g, r]

        return prob_map
    def get_motion_prediction(self, values, max_num_frames):
        pred = np.array([0.0, 0.0])
        if len(values) < 3:
            pred[0] = 0
            pred[1] = 0
        else:
            max_num_frames += 2
            A1 = 0.8
            A2 = -1
            
            V = values[max(0, len(values) - max_num_frames):]
            P = []
            for i in range(2, len(V)):
                P.append(np.array([
                    A1 * (V[i][0] - V[i - 2][0]) + A2 * (V[i - 1][0] - V[i - 2][0]),
                    A1 * (V[i][1] - V[i - 2][1]) + A2 * (V[i - 1][1] - V[i - 2][1])
                ]))
            for p in P:
                pred += p
            pred /= len(P) if P else 1
        return tuple(pred)
    def tracker_dat_update(self, I):
            img_preprocessed = cv2.resize(I, None, fx=self.scale_factor_, fy=self.scale_factor_)
            img = None

            # Color space conversion based on configuration
            if self.cfg['color_space'] == 1:  # RGB
                img = img_preprocessed.copy()
            elif self.cfg['color_space'] == 2:  # LAB
                img = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2Lab)
            elif self.cfg['color_space'] == 3:  # HSV
                img = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2HSV)
            elif self.cfg['color_space'] == 4:  # Grayscale
                img = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2GRAY)
            else:
                print("color_space does not equal any of the above cases")

            prev_pos = self.target_pos_history_[-1] if self.target_pos_history_ else (0, 0)
            prev_sz = self.target_sz_history_[-1] if self.target_sz_history_ else (0, 0)

            if self.cfg['motion_estimation_history_size'] > 0:
                motion_pred = self.get_motion_prediction(self.target_pos_history_, self.cfg['motion_estimation_history_size'])
                prev_pos = (prev_pos[0] + motion_pred[0], prev_pos[1] + motion_pred[1])

            target_pos = (prev_pos[0] * self.scale_factor_, prev_pos[1] * self.scale_factor_)
            target_sz = (prev_sz[0] * self.scale_factor_, prev_sz[1] * self.scale_factor_)

            # Calculate search size
            search_sz = (
                int(target_sz[0] + self.cfg['search_win_padding'] * max(target_sz)),
                int(target_sz[1] + self.cfg['search_win_padding'] * max(target_sz))
            )
            search_rect = self.pos2rect(target_pos, search_sz, img.shape[:2])
            search_win, padded_search_win = self.get_subwindow_masked(img, target_pos, search_sz)

            # Apply probability LUT
            pm_search = self.get_foreground_prob(search_win, self.prob_lut_, self.cfg['bin_mapping'])
            pm_search_dist = None

            if self.cfg['distractor_aware']:
                pm_search_dist = self.get_foreground_prob(search_win, self.prob_lut_distractor_, self.cfg['bin_mapping'])
                pm_search = (pm_search + pm_search_dist) / 2.0
            pm_search[padded_search_win == 0] = 0  # Masking

            # Calculate Cosine / Hanning window
            cos_win = hann(search_sz[0])[:, None] * hann(search_sz[1])

            hypotheses, vote_scores, dist_scores = self.get_nms_rects(pm_search, target_sz, self.cfg['nms_scale'], 
                                                                        self.cfg['nms_overlap'], self.cfg['nms_score_factor'], 
                                                                        cos_win, self.cfg['nms_include_center_vote'])

            candidate_centers = []
            candidate_scores = []
            for i in range(len(hypotheses)):
                center = (float(hypotheses[i][0] + hypotheses[i][2] / 2), float(hypotheses[i][1] + hypotheses[i][3] / 2))
                candidate_centers.append(center)
                candidate_scores.append(vote_scores[i] * dist_scores[i])

            best_candidate_index = np.argmax(candidate_scores)
            target_pos = candidate_centers[best_candidate_index]

            # Distractor processing
            distractors = []
            distractor_overlap = []
            if len(hypotheses) > 1:
                target_rect = self.pos2rect(target_pos, target_sz, pm_search.shape)
                for i in range(len(hypotheses)):
                    if i != best_candidate_index:
                        distractors.append(hypotheses[i])
                        overlap = self.intersection_over_union(target_rect, distractors[-1])
                        distractor_overlap.append(overlap)

            # Localization visualization
            if self.cfg['show_figures']:
                pm_search_color = (pm_search * 255).astype(np.uint8)
                pm_search_color = cv2.applyColorMap(pm_search_color, cv2.COLORMAP_JET)
                for i in range(len(hypotheses)):
                    color = (0, 255, 255 * (i != best_candidate_index))
                    cv2.rectangle(pm_search_color, hypotheses[i], color, 2)
                cv2.imshow("Search Window", pm_search_color)
                cv2.waitKey(1)

            # Appearance update
            target_pos_img = (target_pos[0] + search_rect[0], target_pos[1] + search_rect[1])
            if self.cfg['prob_lut_update_rate'] > 0:
                surr_sz = (
                    int(self.cfg['surr_win_factor'] * target_sz[0]),
                    int(self.cfg['surr_win_factor'] * target_sz[1])
                )
                surr_rect = self.pos2rect(target_pos_img, surr_sz, img.shape[:2])
                obj_rect_surr = self.pos2rect(target_pos_img, target_sz, img.shape[:2])

                obj_rect_surr = (obj_rect_surr[0] - surr_rect[0], obj_rect_surr[1] - surr_rect[1], obj_rect_surr[2], obj_rect_surr[3])
                surr_win = self.get_subwindow_masked(img, target_pos_img, surr_sz)[0]

                prob_lut_bg = self.get_foreground_prob(surr_win, self.prob_lut_, self.cfg['bin_mapping'])

                if self.cfg['distractor_aware']:
                    if len(distractors) > 1:
                        obj_rect = self.pos2rect(target_pos, target_sz, search_win.shape)
                        prob_lut_dist = self.get_foreground_prob(search_win, obj_rect, self.cfg['num_bins'])
                        self.prob_lut_distractor_ = (1 - self.cfg['prob_lut_update_rate']) * self.prob_lut_distractor_ + \
                                                    self.cfg['prob_lut_update_rate'] * prob_lut_dist
                    else:
                        self.prob_lut_distractor_ = (1 - self.cfg['prob_lut_update_rate']) * self.prob_lut_distractor_ + \
                                                    self.cfg['prob_lut_update_rate'] * prob_lut_bg

                    if not distractors or (max(distractor_overlap) < 0.1):
                        self.prob_lut_ = (1 - self.cfg['prob_lut_update_rate']) * self.prob_lut_ + \
                                        self.cfg['prob_lut_update_rate'] * prob_lut_bg
    def intersection_over_union(self, target_rect, candidates):
        intersection_area = (target_rect & candidates).area()
        return float(intersection_area) / float(target_rect.area() + candidates.area() - intersection_area)
    def get_subwindow_masked(self, im, pos, sz):
        xs_1 = int(np.floor(pos[0])) + 1 - int(np.floor(sz[0] / 2.))
        xs_2 = int(np.floor(pos[0])) + sz[0] - int(np.floor(sz[0] / 2.))
        ys_1 = int(np.floor(pos[1])) + 1 - int(np.floor(sz[1] / 2.))
        ys_2 = int(np.floor(pos[1])) + sz[1] - int(np.floor(sz[1] / 2.))

        out = self.get_subwindow(im, pos, sz)

        bbox = (xs_1, ys_1, sz[0], sz[1])
        bbox = (max(bbox[0], 0), max(bbox[1], 0), min(bbox[0] + bbox[2], im.shape[1] - 1), min(bbox[1] + bbox[3], im.shape[0] - 1))
        bbox = (bbox[0] - xs_1, bbox[1] - ys_1, bbox[2], bbox[3])
        
        mask = np.ones(sz, dtype=np.uint8)
        mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 0
        return out, mask



    def get_nms_rects(self, prob_map, obj_sz, scale, overlap, score_frac, dist_map, include_inner):
        height, width = prob_map.shape[:2]
        
        rect_sz = (int(np.floor(obj_sz[0] * scale)), int(np.floor(obj_sz[1] * scale)))
        o_x, o_y = (0, 0)
        
        if include_inner:
            o_x = round(max(1.0, rect_sz[0] * 0.2))
            o_y = round(max(1.0, rect_sz[1] * 0.2))
        
        stepx = max(1, int(round(rect_sz[0] * (1.0 - overlap))))
        stepy = max(1, int(round(rect_sz[1] * (1.0 - overlap))))
        
        posx = list(range(0, width - rect_sz[0] + 1, stepx))
        posy = list(range(0, height - rect_sz[1] + 1, stepy))

        x, y = np.meshgrid(posx, posy, indexing='ij')
        r = np.minimum(x + rect_sz[0], width - 1)
        b = np.minimum(y + rect_sz[1], height - 1)
        
        boxes = [(x[i, j], y[i, j], r[i, j] - x[i, j], b[i, j] - y[i, j]) 
                for i in range(x.shape[0]) for j in range(x.shape[1])]
        
        boxes_inner = []
        
        if include_inner:
            boxes_inner = [(x[i, j] + o_x, y[i, j] + o_y, 
                            r[i, j] - x[i, j] - 2 * o_x, 
                            b[i, j] - y[i, j] - 2 * o_y) 
                        for i in range(x.shape[0]) for j in range(x.shape[1])]
        
        bl = np.stack([x, b], axis=-1)
        br = np.stack([r, b], axis=-1)
        tl = np.stack([x, y], axis=-1)
        tr = np.stack([r, y], axis=-1)
        
        bl_inner, br_inner, tl_inner, tr_inner = (None, None, None, None)
        
        if include_inner:
            bl_inner = np.stack([x + o_x, b - o_y], axis=-1)
            br_inner = np.stack([r - o_x, b - o_y], axis=-1)
            tl_inner = np.stack([x + o_x, y + o_y], axis=-1)
            tr_inner = np.stack([r - o_x, y + o_y], axis=-1)
        
        int_prob_map = cv2.integral(prob_map)
        int_dist_map = cv2.integral(dist_map)
        
        # Ensure these are 2D arrays with valid values
        print("int_prob_map shape:", int_prob_map.shape)
        print("int_dist_map shape:", int_dist_map.shape)

        v_scores = np.zeros(len(bl), dtype=np.float32)
        d_scores = np.zeros(len(bl), dtype=np.float32)
        
        for i in range(len(bl)):
            br_idx = br[i]
            bl_idx = bl[i]
            tr_idx = tr[i]
            tl_idx = tl[i]

            print(f'Indices: bl={bl_idx}, br={br_idx}, tl={tl_idx}, tr={tr_idx}')
            
            # Extract scalar values and ensure they are scalars
            prob_br = int_prob_map[br_idx[1], br_idx[0]]
            prob_bl = int_prob_map[bl_idx[1], bl_idx[0]]
            prob_tr = int_prob_map[tr_idx[1], tr_idx[0]]
            prob_tl = int_prob_map[tl_idx[1], tl_idx[0]]
            
            # Debugging: Ensure extracted values are scalars
            if np.ndim(prob_br) != 0 or np.ndim(prob_bl) != 0 or np.ndim(prob_tr) != 0 or np.ndim(prob_tl) != 0:
                print("Error: One of the values is not a scalar")
                print(f"prob_br: {prob_br}, prob_bl: {prob_bl}, prob_tr: {prob_tr}, prob_tl: {prob_tl}")

            # Assign the scores
            v_scores[i] = float(prob_br - prob_bl - prob_tr + prob_tl)  # Ensure it's a float
            dist_br = int_dist_map[br_idx[1], br_idx[0]]
            dist_bl = int_dist_map[bl_idx[1], bl_idx[0]]
            dist_tr = int_dist_map[tr_idx[1], tr_idx[0]]
            dist_tl = int_dist_map[tl_idx[1], tl_idx[0]]
            
            d_scores[i] = float(dist_br - dist_bl - dist_tr + dist_tl)  # Ensure it's a float

        scores_inner = np.zeros(len(bl), dtype=np.float32)
        
        if include_inner:
            for i in range(len(bl)):
                scores_inner[i] = (int_prob_map[br_inner[i][1], br_inner[i][0]] - 
                                int_prob_map[bl_inner[i][1], bl_inner[i][0]] - 
                                int_prob_map[tr_inner[i][1], tr_inner[i][0]] + 
                                int_prob_map[tl_inner[i][1], tl_inner[i][0]])

                if (rect_sz[0] - 2 * o_x) > 0 and (rect_sz[1] - 2 * o_y) > 0:
                    v_scores[i] += (scores_inner[i] / float((rect_sz[0] - 2 * o_x) * (rect_sz[1] - 2 * o_y)))

        top_rects = []
        top_vote_scores = []
        top_dist_scores = []
        
        max_idx = np.argmax(v_scores)
        max_score = v_scores[max_idx]
        
        while max_score > score_frac * max(v_scores):
            x, y, w, h = boxes[max_idx]
            prob_map[y:y + h, x:x + w] = 0.0
            
            top_rects.append(boxes[max_idx])
            top_vote_scores.append(v_scores[max_idx])
            top_dist_scores.append(d_scores[max_idx])
            
            boxes.pop(max_idx)
            if include_inner:
                boxes_inner.pop(max_idx)
            
            bl = np.delete(bl, max_idx, axis=0)
            br = np.delete(br, max_idx, axis=0)
            tl = np.delete(tl, max_idx, axis=0)
            tr = np.delete(tr, max_idx, axis=0)
            
            if include_inner:
                bl_inner = np.delete(bl_inner, max_idx, axis=0)
                br_inner = np.delete(br_inner, max_idx, axis=0)
                tl_inner = np.delete(tl_inner, max_idx, axis=0)
                tr_inner = np.delete(tr_inner, max_idx, axis=0)
            
            int_prob_map = cv2.integral(prob_map)
            int_dist_map = cv2.integral(dist_map)
            
            v_scores = np.zeros(len(bl), dtype=np.float32)
            d_scores = np.zeros(len(bl), dtype=np.float32)
            
            for i in range(len(bl)):
                v_scores[i] = (int_prob_map[br[i][1], br[i][0]] - 
                            int_prob_map[bl[i][1], bl[i][0]] - 
                            int_prob_map[tr[i][1], tr[i][0]] + 
                            int_prob_map[tl[i][1], tl[i][0]])
                d_scores[i] = (int_dist_map[br[i][1], br[i][0]] - 
                            int_dist_map[bl[i][1], bl[i][0]] - 
                            int_dist_map[tr[i][1], tr[i][0]] + 
                            int_dist_map[tl[i][1], tl[i][0]])
            
            if include_inner:
                for i in range(len(bl)):
                    scores_inner[i] = (int_prob_map[br_inner[i][1], br_inner[i][0]] - 
                                    int_prob_map[bl_inner[i][1], bl_inner[i][0]] - 
                                    int_prob_map[tr_inner[i][1], tr_inner[i][0]] + 
                                    int_prob_map[tl_inner[i][1], tl_inner[i][0]])
                    v_scores[i] += (scores_inner[i] / float(rect_sz[0] * rect_sz[1]))

            max_idx = np.argmax(v_scores)
            max_score = v_scores[max_idx]

        return top_rects, top_vote_scores, top_dist_scores
