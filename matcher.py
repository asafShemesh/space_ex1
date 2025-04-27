import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
from scipy.spatial import cKDTree

def detect_stars(image_path, min_brightness=80):

    start_time = time.time()

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Apply Gaussian Blur
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 100
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.25
    params.filterByColor = True
    params.blobColor = 255
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image_blur)

    stars = []
    for keypoint in keypoints:
        cX = int(keypoint.pt[0])
        cY = int(keypoint.pt[1])
        radius = int(keypoint.size / 2)

        # Brightness estimation
        r = int(max(5, radius))
        x1 = max(cX - r, 0)
        x2 = min(cX + r, image.shape[1] - 1)
        y1 = max(cY - r, 0)
        y2 = min(cY + r, image.shape[0] - 1)
        star_region = image[y1:y2, x1:x2]
        brightness = np.mean(star_region)

        if brightness < min_brightness:
            continue

        stars.append((cX, cY, radius, brightness))

    end_time = time.time()
    print(f"✅ Detected {len(stars)} stars in ({end_time - start_time:.2f} seconds)")
    return stars


def find_top_bright_triplets(stars, top_brightest=15, top_triplets=5, min_area=50):
    sorted_stars = sorted(enumerate(stars), key=lambda x: -x[1][3])
    selected = sorted_stars[:top_brightest]
    indices = [idx for idx, _ in selected]
    points = np.array([[star[0], star[1]] for _, star in selected])

    triplet_candidates = []
    for (i, j, k) in itertools.combinations(range(len(points)), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs(
            p1[0]*(p2[1]-p3[1]) +
            p2[0]*(p3[1]-p1[1]) +
            p3[0]*(p1[1]-p2[1])
        )
        if area > min_area:
            triplet_candidates.append((area, (indices[i], indices[j], indices[k])))

    triplet_candidates.sort(key=lambda x: x[0], reverse=True)
    return [trip for _, trip in triplet_candidates[:top_triplets]]




def match_fixed_triplet(stars1, stars2, p1_idx, p2_idx, p3_idx, distance_thresh=10):

    pts1 = np.array([[x,y] for x,y,_,_ in stars1], dtype=np.float32)
    pts2 = np.array([[x,y] for x,y,_,_ in stars2], dtype=np.float32)
    tree2 = cKDTree(pts2)

    src_pts = pts1[[p1_idx, p2_idx, p3_idx]]

    best_count = 0
    best_mean_dist = float('inf')
    best_match = None

    for combo in itertools.combinations(range(len(stars2)), 3):
        dst_pts = pts2[list(combo)]
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if M is None:
            continue

        # Warp all pts1
        ones = np.ones((len(pts1),1), dtype=np.float32)
        pts1_h = np.hstack([pts1, ones])
        warped = (M.dot(pts1_h.T)).T  # shape (N1,2)

        # Find all potential pairs within threshold
        dists, idxs = tree2.query(warped, distance_upper_bound=distance_thresh)
        candidates = [
            (i1, i2, d) 
            for i1, (i2, d) in enumerate(zip(idxs, dists))
            if d < distance_thresh
        ]
        # sort by distance
        candidates.sort(key=lambda x: x[2])

        # greedy one-to-one assignment
        used2 = set()
        unique_pairs = []
        for i1, i2, d in candidates:
            if i2 not in used2:
                used2.add(i2)
                unique_pairs.append((i1, i2, d))

        count = len(unique_pairs)
        if count > 0:
            mean_dist = float(np.mean([d for _,_,d in unique_pairs]))
        else:
            mean_dist = float('inf')

        if (count > best_count) or (count == best_count and mean_dist < best_mean_dist):
            best_count = count
            best_mean_dist = mean_dist
            best_match = {
                'matched_triplet2': combo,
                'affine_matrix': M,
                'mean_distance': mean_dist,
                # drop the distance from the final list if you only want indices
                'matched_indices': [(i1,i2) for i1,i2,_ in unique_pairs]
            }

    return best_count, best_match


if __name__ == "__main__":
    # Paths
    image_path1 = r"C:\space_ex1\IMG_3046.jpg"
    image_path2 = r"C:\space_ex1\IMG_3048.jpg"

    # Detect
    stars_list1 = detect_stars(image_path1)
    stars_list2 = detect_stars(image_path2)

    # Candidate triplets
    top_triplets = find_top_bright_triplets(stars_list1,top_brightest=15,top_triplets=5,min_area=50)

    # Find best overall
    best_count = 0
    best_match = None
    for p1,p2,p3 in top_triplets:
        cnt, match = match_fixed_triplet(stars_list1, stars_list2, p1,p2,p3, distance_thresh=40)
        if match and cnt > best_count:
            best_count = cnt
            best_match = match

    print(f"pairs matched = {best_count}")

    # --- Visualization ---
    if best_match:
        # Load color images
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        h1,w1 = img1.shape[:2]
        h2,w2 = img2.shape[:2]

        # Build canvas
        canvas = np.zeros((max(h1,h2), w1+w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2

        # Draw lines
        pairs = best_match['matched_indices']
        for (i1,i2) in pairs:
            x1,y1,_,_ = stars_list1[i1]
            x2,y2,_,_ = stars_list2[i2]
            color = tuple(np.random.randint(0,255,3).tolist())
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2)+w1, int(y2))
            cv2.line(canvas, pt1, pt2, color, thickness=2)
            # Optionally draw circles too:
            cv2.circle(canvas, pt1, 55, color, 6)
            cv2.circle(canvas, pt2, 55, color, 6)

        # Show
        plt.figure(figsize=(12,6))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"{len(pairs)} matches were found")
        plt.show()
