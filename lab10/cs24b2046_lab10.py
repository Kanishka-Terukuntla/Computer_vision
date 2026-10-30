
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Bokeh Effect Tool",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Bokeh Effect Tool")
st.write(
    "Upload an original image and its depth map. "
    "Keep the foreground sharp and blur the background "
    "using depth information."
)


@st.cache_resource(show_spinner="Loading depth estimation model...")
def load_depth_model():
    from transformers import pipeline

    return pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )


def read_rgb(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.array(image)


def estimate_depth(rgb):
    model = load_depth_model()
    result = model(Image.fromarray(rgb))
    depth = np.array(result["depth"].convert("L"))
    return depth


def normalize_depth(depth):
    depth = depth.astype(np.float32)
    low, high = np.percentile(depth, [1, 99])

    if high <= low:
        return np.full(depth.shape, 0.5, dtype=np.float32)

    depth = (depth - low) / (high - low)
    return np.clip(depth, 0.0, 1.0)


def make_bokeh(rgb, depth, threshold, blur_strength,
               feather, invert_depth):
    # Normalize depth to [0, 1].
    depth = normalize_depth(depth)

    # Convention used here: white = near, black = far.
    if invert_depth:
        depth = 1.0 - depth

    # Resize depth map to match the original image.
    h, w = rgb.shape[:2]
    depth = cv2.resize(
        depth, (w, h), interpolation=cv2.INTER_LINEAR
    )

    # Slight smoothing reduces noise in the depth map.
    depth = cv2.GaussianBlur(depth, (5, 5), 0)

    threshold = threshold / 100.0
    feather = max(feather / 100.0, 0.005)

    # Foreground mask: 1 = sharp, 0 = blurred.
    x = np.clip(
        (depth - (threshold - feather)) / (2 * feather),
        0.0,
        1.0,
    )
    foreground = x * x * (3.0 - 2.0 * x)
    foreground = cv2.GaussianBlur(
        foreground, (0, 0), sigmaX=1.2
    )
    foreground = np.clip(foreground, 0.0, 1.0)

    # Distance from foreground controls blur intensity.
    far_amount = np.clip(1.0 - depth, 0.0, 1.0)

    original = rgb.astype(np.float32)
    result = original.copy()

    # Create progressively stronger blur layers.
    # Pixels farther away receive stronger blur.
    max_sigma = float(blur_strength)

    levels = [
        (0.25, max_sigma * 0.20),
        (0.50, max_sigma * 0.40),
        (0.75, max_sigma * 0.70),
        (1.01, max_sigma),
    ]

    previous = 0.0
    blurred_background = np.zeros_like(original)

    for upper, sigma in levels:
        layer = cv2.GaussianBlur(
            rgb,
            (0, 0),
            sigmaX=max(sigma, 0.1),
            sigmaY=max(sigma, 0.1),
        ).astype(np.float32)

        # Soft bands avoid abrupt transitions between blur levels.
        band = np.clip(
            (far_amount - previous) / max(upper - previous, 1e-6),
            0.0,
            1.0,
        )

        # Accumulate a depth-dependent background blur.
        blurred_background = (
            blurred_background * (1.0 - band[..., None])
            + layer * band[..., None]
        )

        previous = upper

    # Composite: retain original foreground, blur background.
    result = (
        original * foreground[..., None]
        + blurred_background * (1.0 - foreground[..., None])
    )

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result, depth, foreground


with st.sidebar:
    st.header("Image Inputs")

    rgb_file = st.file_uploader(
        "1. Original RGB image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    automatic_depth = st.checkbox(
        "Generate depth automatically",
        value=False,
        help="Uses a pretrained depth-estimation model.",
    )

    depth_file = None

    if not automatic_depth:
        depth_file = st.file_uploader(
            "2. Corresponding depth map",
            type=["jpg", "jpeg", "png", "webp"],
        )

    st.divider()
    st.header("Bokeh Settings")

    blur_strength = st.slider(
        "Blur intensity",
        min_value=1,
        max_value=50,
        value=15,
    )

    threshold = st.slider(
        "Foreground threshold (%)",
        min_value=0,
        max_value=100,
        value=50,
    )

    feather = st.slider(
        "Foreground edge softness (%)",
        min_value=1,
        max_value=30,
        value=10,
    )

    invert_depth = st.checkbox(
        "Invert depth map",
        value=False,
        help=(
            "Enable if the depth map has white far away "
            "and black nearby."
        ),
    )

    generate = st.button(
        "Generate Bokeh Image",
        type="primary",
        use_container_width=True,
    )


if rgb_file is None:
    st.info(
        "Upload an RGB image in the sidebar to get started. "
        "You can use the provided sample image."
    )
    st.stop()

rgb = read_rgb(rgb_file)

if generate:
    if not automatic_depth and depth_file is None:
        st.error(
            "Upload a depth map or enable automatic depth estimation."
        )
        st.stop()

    try:
        with st.spinner("Generating the bokeh effect..."):
            if automatic_depth:
                depth = estimate_depth(rgb)
            else:
                depth_image = Image.open(depth_file)
                depth_image = ImageOps.exif_transpose(
                    depth_image
                ).convert("L")
                depth = np.array(depth_image)

            output, normalized_depth, foreground = make_bokeh(
                rgb=rgb,
                depth=depth,
                threshold=threshold,
                blur_strength=blur_strength,
                feather=feather,
                invert_depth=invert_depth,
            )

        st.success("Bokeh image generated successfully!")

        tab_result, tab_inputs, tab_masks = st.tabs(
            ["Bokeh Result", "Input Images", "Depth & Mask"]
        )

        with tab_result:
            st.subheader("Final Bokeh Image")
            st.image(output, use_container_width=True)

            buffer = io.BytesIO()
            Image.fromarray(output).save(buffer, format="PNG")

            st.download_button(
                "Download Bokeh Image",
                data=buffer.getvalue(),
                file_name="bokeh_output.png",
                mime="image/png",
            )

        with tab_inputs:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original RGB")
                st.image(rgb, use_container_width=True)
            with col2:
                st.subheader("Depth Map")
                st.image(
                    (normalized_depth * 255).astype(np.uint8),
                    use_container_width=True,
                )

        with tab_masks:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Estimated Depth")
                st.image(
                    (normalized_depth * 255).astype(np.uint8),
                    use_container_width=True,
                )
            with col2:
                st.subheader("Foreground Mask")
                st.image(
                    (foreground * 255).astype(np.uint8),
                    use_container_width=True,
                )

    except Exception as exc:
        st.error(f"Could not generate the image: {exc}")
        st.info(
            "Check the image files and depth-map polarity. "
            "For automatic depth, install all requirements "
            "and ensure the model can be downloaded."
        )
else:
    st.image(rgb, caption="Original image", use_container_width=True)
    st.caption(
        "Adjust the settings in the sidebar, then click "
        "'Generate Bokeh Image'."
    )
