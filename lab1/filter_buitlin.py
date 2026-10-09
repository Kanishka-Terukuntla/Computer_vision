
import cv2
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_IMAGES = [
    "image1.png",
    "image2.png"
]

KERNEL_SIZE = 3


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

os.makedirs("output_builtin", exist_ok=True)


# ---------------------------------------------------------
# Process each image
# ---------------------------------------------------------

for image_path in INPUT_IMAGES:

    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not read", image_path)
        continue

    # -----------------------------------------------------
    # 1. Mean Filter
    # -----------------------------------------------------
    mean_image = cv2.blur(
        image,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    # -----------------------------------------------------
    # 2. Median Filter
    # -----------------------------------------------------
    median_image = cv2.medianBlur(
        image,
        KERNEL_SIZE
    )

    # -----------------------------------------------------
    # 3. Mean followed by Median
    # -----------------------------------------------------
    mean_median_image = cv2.medianBlur(
        mean_image,
        KERNEL_SIZE
    )

    # -----------------------------------------------------
    # 4. Median followed by Mean
    # -----------------------------------------------------
    median_mean_image = cv2.blur(
        median_image,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    cv2.imwrite(
        f"output_builtin/{name}_mean.png",
        mean_image
    )

    cv2.imwrite(
        f"output_builtin/{name}_median.png",
        median_image
    )

    cv2.imwrite(
        f"output_builtin/{name}_mean_median.png",
        mean_median_image
    )

    cv2.imwrite(
        f"output_builtin/{name}_median_mean.png",
        median_mean_image
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(mean_image, cmap="gray")
    plt.title("Mean Filter")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(median_image, cmap="gray")
    plt.title("Median Filter")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(mean_median_image, cmap="gray")
    plt.title("Mean → Median")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(median_mean_image, cmap="gray")
    plt.title("Median → Mean")
    plt.axis("off")

    plt.suptitle(name)

    plt.tight_layout()
    plt.show()

    print(f"Processed: {image_path}")