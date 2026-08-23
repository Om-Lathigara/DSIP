import cv2
import numpy as np
from skimage.exposure import match_histograms
import os

image_paths = [
    r"D:\MARWADI\YEAR3\SEM5\DSIP\OPE8\Screenshot 2026-08-23 194941.png",
    r"D:\MARWADI\YEAR3\SEM5\DSIP\OPE8\Screenshot 2026-08-23 195017.png",
    r"D:\MARWADI\YEAR3\SEM5\DSIP\OPE8\Screenshot 2026-08-23 195027.png",
    r"D:\MARWADI\YEAR3\SEM5\DSIP\OPE8\Screenshot 2026-08-23 195033.png",
    r"D:\MARWADI\YEAR3\SEM5\DSIP\OPE8\Screenshot 2026-08-23 195042.png"
]

reference_path = r"D:\MARWADI\YEAR3\SEM5\DSIP\OPE8\reference.jpg"

output_folder = r"D:\MARWADI\YEAR3\SEM5\DSIP\OPE8"

os.makedirs(output_folder, exist_ok=True)

reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

if reference is None:
    raise FileNotFoundError("Could not read reference image")

for i, path in enumerate(image_paths, 1):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    equalized = cv2.equalizeHist(img)

    matched = match_histograms(img, reference)
    matched = np.clip(matched, 0, 255).astype(np.uint8)

    cv2.imwrite(
        os.path.join(output_folder, f"image{i}_equalized.png"),
        equalized
    )

    cv2.imwrite(
        os.path.join(output_folder, f"image{i}_matched.png"),
        matched
    )

    print(f"Image {i} processed")

print("All five images processed.")