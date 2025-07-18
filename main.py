import cv2
import numpy as np
import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

def describe_image(img):
    # Getting greyscale to make calculations faster
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ensure effect of different lighting per frames is minimal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_normalised = clahe.apply(img_grey)
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(img_normalised, None)


def get_matches(des1, des2):
    # Using brute force matcher for simple speed when working
    # with so many images
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    # 0.5 from https://andrewdcampbell.github.io/stitching-photo-mosaics
    good_matches = [m1 for m1, m2 in matches if m1.distance < 0.5 * m2.distance]    
    return good_matches


# Reference function for simple frame extraction
# with stride
def extract_keyframes_naive(video_path):
    frames = list()
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    _, last_frame = cap.read()
    frames.append(last_frame)
    # skip a certain amount of frames to speed up computation
    frameskip = int(fps / 1.5)
    i = 0
    while True:
        ret, last_frame = cap.read()
        if not ret:
            break
        if i == frameskip:
            frames.append(last_frame)
            i = 0
        i+=1
    print(f"frame num: {len(frames)}")
    return frames


# Intelligent frame extraction using feature threshold
def extract_keyframes(video_path):
    frames = list()
    cap = cv2.VideoCapture(video_path)
    _, last_frame = cap.read()
    frames.append(last_frame)
    _, des_lst = describe_image(last_frame)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frame_count)):
        ret, current_frame = cap.read()
        if not ret:
            frames.append(last_frame)
            break
        # 3 Chosen based on trial and error
        if i % 3 == 0:
            _, des_crt = describe_image(current_frame)
            matches = get_matches(des_lst, des_crt)
            # Only append if below a certain threshold to avoid appending
            # frames too close to each other
            if len(matches) > 50 and len(matches) < 300:
                frames.append(current_frame)
                last_frame = current_frame
                des_lst = des_crt

    print(f"frame num: {len(frames)}")
    return frames


def get_homography(matches, kp_1, kp_2):
    # Storing coordinates of points corresponding to the matches found in both the images
    src_pts = np.float32([kp_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if len(src_pts) < 4 or len(dst_pts) < 4:
        return None
    
    # Finding the homography matrix(transformation matrix).
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H


# Inspired by:
# https://github.com/KEDIARAHUL135/PanoramaStitching/blob/master/main.py
# Used instead of having a estimated offset range as in lab 4
# Calculates offset based on the image coordinates after transformation with the homography matrix
def get_modified_size_and_homography(H, img_base_shape, img_add_shape):
    h1, w1 = img_base_shape
    h2, w2 = img_add_shape
    
    # Storing the initial 4 corners of the image
    # In order of: TL, TR, BR, BL
    initial_image_corners = np.float32([[0,      0     ],
                                        [w2 - 1, 0     ],
                                        [w2 - 1, h2 - 1],
                                        [0,      h2 - 1]])

    # In Homogenous coordinates so can transform with homography matrix
    initial_points_homogenous = np.vstack((initial_image_corners.T, np.array([1, 1, 1, 1])))
     
    # Transforming with the homography matrix to 
    # get the coordinates of new extents of the image
    final_points = np.dot(H, initial_points_homogenous)

    # Get the rows in an easily workable form
    [x, y, z] = final_points
    # Get normalised coordinates
    x = np.divide(x, z)
    y = np.divide(y, z)

    # Get the extents of the transformed image
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    # Calculate new image offset to ensure no cropping of image
    ## Following cv2 (y, x)
    offset = [abs(min_y) if min_y < 0 else 0, abs(min_x) if min_x < 0 else 0]

    # Ensuring that the image fits within the frame
    x += offset[1]
    y += offset[0]

    # Get the correct representation of the new coordinates
    new_points = np.float32(np.array([x, y]).T)

    ## Recalculate homography from the newly calculated points and the original position
    H = cv2.getPerspectiveTransform(initial_image_corners, new_points)

    # Get whatever sizes are needed to fit the image without cropping
    new_width  = max(max_x - min_x, w1 + abs(min_x))
    new_height = max(max_y - min_y, h1 + abs(min_y))
    
    return [new_width, new_height], offset, H

# Function to roughly estimate the focal length of 
# the camera used to capture the images
## kinda bad should improve
def estimate_focal_length(images):
    focal_lengths = []
    for i in range(0, len(images) - 1):
        kp_1, des_1 = describe_image(images[i])
        kp_2, des_2 = describe_image(images[i + 1])

        # Get the fundamental matrix from the the two images
        matches = get_matches(des_1, des_2)
        src_pts = np.float32([kp_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        fundamental_matrix, _ = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # Calculate the epipolar lines
        lines = cv2.computeCorrespondEpilines(dst_pts, 2, fundamental_matrix)
        lines = lines.reshape(-1, 3) # squeeze down into a 1d array

        # Estimate focal length from the z distances
        for line in lines:
            focal_lengths.append(abs(line[2]))

    ## To be honest this is very rough and not great
    ## But it works ok for now
    # Return the average of all focal lengths
    # Multiply by constant to decrease distortion (kinda just trial and error tbh)
    return np.mean(focal_lengths) * np.log(np.pi * np.pi)


def match_extents(img):
    # First, ensure that canvas extents matches image extents
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_black_indices = np.where(grey > 0)
    # Get the minimum and maximum coordinates of non-black pixels
    min_x, min_y = np.min(non_black_indices[1]), np.min(non_black_indices[0])
    max_x, max_y = np.max(non_black_indices[1]), np.max(non_black_indices[0])
    # Crop canvas to size
    cropped = img[min_y:max_y, min_x:max_x]
    return cropped, min_x, max_x, min_y, max_y
    

def crop_first_column(img, min_x, max_x, min_y):
    first_column = img[:, 0, :]
    # Find the top-most non-black pixel and bottom-most non-black pixel
    # Due to mask transformation, left hand side of image will always be flush with the x axis
    top_pixel = None
    bottom_pixel = None
    for i in range(len(first_column)):
        if not np.all(first_column[i] == [0, 0, 0]):
            if top_pixel is None:
                top_pixel = min_y + i
            bottom_pixel = min_y + i
    cropped = img[top_pixel:bottom_pixel, min_x:max_x]
    return cropped

def check_row(img, width, index):
    for x in range(width):
        if img[index, x] == 0:
            return True
    return False

def check_column(img, height, index):
    for y in range(height):
        if img[y, index] == 0:
            return True
    return False

## Gets the used extent of the image
## Crops so that no black areas are shown
def crop_image(img):
    cropped, min_x, max_x, min_y, _ = match_extents(img)

    #### IF YOU WANT TO SEE THE DARK IMAGE RESULTS, UNCOMMENT THE FOLLOWING ####
    #return cropped
    cropped = crop_first_column(cropped, min_x, max_x, min_y)

    grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape[:2]

    # Check first row for background pixels
    crop_top = check_row(grey, width, 0)
    # Check bottom row for background pixels
    crop_bottom = check_row(grey, width, height - 1)
    # Check right column for background pixels
    crop_right = check_column(grey, height, width - 1)

    # Now iteratively start reducing ROI size until
    # no more border remains in the image
    ## Not super great, doesn't return maximum image ROI, but works
    y_start = 0
    y_end, x_end = cropped.shape[:2]

    has_black = np.any(grey == 0)
    while has_black:
        # Only reduce ROI on sections with background
        if crop_top:
            y_start+=1
        if crop_bottom:
            y_end-=1
        if crop_right:
            x_end-=1
        temp = cropped[y_start: y_end, 0 :x_end]
        grey = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        has_black = np.any(grey == 0)
    cropped = temp

    return cropped


# Blend the original image and the addition image togther
def blend_images(img_base, img_add, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.bitwise_not(mask)
    kernel = np.ones((5,5))
    ## Try and smooth transition areas
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.erode(mask, kernel, iterations=4)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    non_overlapping_region = cv2.bitwise_and(img_base, mask)
    blended = cv2.bitwise_or(img_add, non_overlapping_region)
    return blended


def bilinear_interpolation(x_coord_t, y_coord_t, x_t, y_t, img):
    x_t_int = x_t.astype(int)
    y_t_int = y_t.astype(int)

    # Using same implementation of bilinear interpolation as shown in lecture slides
    # first computing a and b
    a = x_t - x_t_int
    b = y_t - y_t_int

    # Simply working out the interpolation within all four corners
    # of the image
    weight_tl = (1.0 - a) * (1.0 - b)
    weight_tr = (a) * (1.0 - b)
    weight_br = (a) * (b)      
    weight_bl = (1.0 - a) * (b)      
    
    transformed = np.zeros(img.shape, dtype=np.uint8)
    # Apply the appropriate weightings to each section of the image
    # per the bilinear interpolation formula
    transformed[y_coord_t, x_coord_t, :] = (weight_tl[:, np.newaxis] * img[y_t_int,     x_t_int,     :]) + \
                                           (weight_tr[:, np.newaxis] * img[y_t_int,     x_t_int + 1, :]) + \
                                           (weight_br[:, np.newaxis] * img[y_t_int + 1, x_t_int + 1, :]) + \
                                           (weight_bl[:, np.newaxis] * img[y_t_int + 1, x_t_int,     :])
    
    return transformed


## Getting transformed coordinates on cylindrical plane
## Uses the following standard equations:
## Angle:
##      theta = (x - center_x) / f
## Projection:
##      X' = sin(theta)
##      Y' = (y - center_y) / f
##      Z' = cos(theta)
## Unrolling:
##      xT = f X' / Z' + center_x
##      yT = f Y' / Z' + center_y
def project_xy(w, h, x_coord_t, y_coord_t, f):
    theta = (x_coord_t - w // 2) / f
    x_dash = np.sin(theta)
    y_dash = (y_coord_t - h // 2) / f
    z_dash = np.cos(theta)

    x_t = f * x_dash / z_dash + w // 2
    y_t = f * y_dash / z_dash + h // 2

    return x_t, y_t

# Inspired by cylindrical projection code from this document:
# https://www.scribd.com/document/510892625/Panorama-Stitching-P2
def cylinder_project(img, f):
    h, w = img.shape[:2]

    # Generate coordinates for the transformed image
    y_coord_t, x_coord_t = np.indices((h, w))
    x_t, y_t = project_xy(w, h, x_coord_t, y_coord_t, f)

    # Integer represenation for proper indexing
    x_t_int = x_t.astype(int)
    y_t_int = y_t.astype(int)

    # Find transformed image points within the initial image
    inside_mask = (x_t_int >= 0) & (x_t_int < w - 1) & (y_t_int >= 0) & (y_t_int < h - 1)

    # Apply mask to keep only inside points
    # Avoids negative indices
    x_coord_t, y_coord_t    = x_coord_t[inside_mask], y_coord_t[inside_mask]
    x_t, y_t                = x_t[inside_mask], y_t[inside_mask]
    x_t_int, y_t_int        = x_t_int[inside_mask], y_t_int[inside_mask]

    transformed = bilinear_interpolation(x_coord_t, y_coord_t, x_t, y_t, img)

    # Get image extent to crop black border
    # Min y is always 0 so don't bother
    min_x = min(x_coord_t)

    # Crops out any black region from the image
    # Cylindrical projection should be symettrical so do the same for both sides
    transformed = transformed[:, min_x : -min_x, :]

    ## Using masks to avoid copying non-image pixels to the panorama
    ## Ensures straight image edges at the x extents too
    mask_x = x_coord_t-min_x
    mask_y = y_coord_t

    img_add_mask = np.zeros(transformed.shape, dtype=np.uint8)
    img_add_mask[mask_y, mask_x, :] = 255

    return transformed, img_add_mask


def stitch_images(images):
    #focal_length = estimate_focal_length(images)
    focal_length = 758.6 ## Using known focal length for my camera
    stitched_image, _ = cylinder_project(images[0], focal_length)
    for i in range(1, len(images)):
        img_add = images[i]
        img_add_cyl, img_add_mask = cylinder_project(img_add, focal_length)
        
        kp_1, des_1 = describe_image(stitched_image)
        kp_2, des_2 = describe_image(img_add_cyl)

        # Finding matches between the 2 images and their keypoints
        matches = get_matches(des_1, des_2)

        # Finding initial homography matrix
        H = get_homography(matches, kp_1, kp_2)
        if H is None:
            break
        
        # Get the new size of image to avoid cropping
        # Recalculate homography with corrected coordinates
        stitched_h, stitched_w = stitched_image.shape[:2]
        new_size, offset, H = get_modified_size_and_homography(H, (stitched_h, stitched_w), img_add_cyl.shape[:2])

        # basic image combination
        # should use something like gain compensation or multi-level banding
        image_add_trans = cv2.warpPerspective(img_add_cyl, H, new_size) 
        img_add_mask_trans = cv2.warpPerspective(img_add_mask, H, new_size) 
        base_image_trans = np.zeros((new_size[1], new_size[0], 3), dtype=np.uint8)
        base_image_trans[offset[0]: offset[0] + stitched_h, offset[1]: offset[1] + stitched_w] = stitched_image
        stitched_image = blend_images(base_image_trans, image_add_trans, img_add_mask_trans)

    stitched_image = crop_image(stitched_image)
    return stitched_image


## Class used for UI
class AppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Panorama Generator")

        # Dropdown list of videos
        self.video_options = ["test/test1_success.mp4", "test/test2_success.mp4", "test/test3_partial.mp4",
                              "test/test4_success.mp4", "test/test5_partial.mp4", "test/test6_partial.mp4", 
                              "test/test7_success.mp4", "test/test8_partial.mp4", "test/test9_failed.mp4",
                              "test/test10_failed.mp4", "test/test11_failed.mp4"]
        self.selected_video = tk.StringVar()
        self.video_dropdown = ttk.Combobox(self.root, textvariable=self.selected_video, values=self.video_options)
        self.video_dropdown.pack()

        # Video player
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack()
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Generate button
        self.generate_button = tk.Button(self.root, text="Generate Panorama", command=self.generate_panorama)
        self.generate_button.pack()

        # Image display for generated panorama
        self.panorama_image_label = tk.Label(self.root)
        self.panorama_image_label.pack()

        # Bind selection change to play video
        self.video_dropdown.bind("<<ComboboxSelected>>", self.play_video)
    
    # Simply convert opencv format to tkinter format image
    def cv_to_tk_image(self, cv_image):
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(image=Image.fromarray(image))
        return image

    # Play the video on the ui
    def play_video(self, event=None):
        video_path = self.selected_video.get()
        if video_path:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image = self.cv_to_tk_image(frame)
                self.video_label.config(image=image)
                self.video_label.image = image
                self.root.update() # Ensure UI doesnt hang
                self.root.after(10) # Play at correct speed

            cap.release()

    # Make the output panorama fit ui better
    def resize_image_display(self, img, max_width=1080):
        original_height, original_width = img.shape[:2]
        scale_factor = max_width / original_width
        new_height = int(original_height * scale_factor)
        resized_image = cv2.resize(img, (max_width, new_height))
        return resized_image
    
    # Run panorama generation in background thread to avoid hanging
    def generate_panorama_thread(self):
        video_path = self.selected_video.get()
        if video_path:
            images = extract_keyframes_naive(video_path)
            panorama_image = stitch_images(images)
            time = datetime.datetime.now()
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(f"results/test_{time_str}.png", panorama_image)
            resized_image = self.resize_image_display(panorama_image)
            image = self.cv_to_tk_image(resized_image)
            self.root.after(0, lambda: self.update_ui(image))  # Schedule UI update in the main thread

    # Callback after finished
    def update_ui(self, image):
        self.panorama_image_label.config(image=image)
        self.panorama_image_label.image = image

    def generate_panorama(self):
        # Start the generate_panorama_thread function on a separate thread
        # avoids hanging of main programs
        threading.Thread(target=self.generate_panorama_thread, daemon=True).start()


if __name__ == '__main__':
    root = tk.Tk()
    app = AppUI(root)
    root.mainloop()