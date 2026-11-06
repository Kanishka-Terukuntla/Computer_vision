# Lab 11 — SIFT Face Recognition Studio

Feature-rich Streamlit application for the Computer Vision Assignment 11.

## Folder structure
```text
lab11_pro/
├── dataset/
│   ├── train/Person_01 ... Person_10/
│   └── test/Person_01 ... Person_10/
├── src/app.py
├── requirements.txt
└── README.md
```

The ZIP includes empty person folders. Copy your own images into them:
- 10 training images per person under `dataset/train/Person_XX/`
- 2 different test images per person under `dataset/test/Person_XX/`

Use the same folder label for the same individual in both splits. Image filenames can remain unchanged.

## Install (Windows)
Open a terminal in the extracted project root:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run
From the project root:
```powershell
streamlit run src/app.py
```

## Features
- Dataset manager: upload multiple images or capture using a camera; browse counts/previews and delete a person's images.
- Face preprocessing: Haar Cascade face detection, crop, grayscale, resize and CLAHE.
- SIFT training: extracts and saves descriptors in `sift_model.pkl`.
- Keypoint visualization: inspect detected SIFT keypoints.
- Recognition: upload or camera input, adjustable Lowe ratio and minimum match threshold.
- Evaluation: test-set accuracy, individual predictions, confusion matrix, image previews and CSV export.
- Explanation tab describing the method.

## Notes
- Install `opencv-contrib-python` because it includes SIFT. Do not install both `opencv-python` and `opencv-contrib-python` in the same environment.
- After changing or adding training images, click Train / Retrain model again.
- If source photos are already tightly cropped faces, leave "Skip image if Haar detector finds no face" unchecked.
- Haar face detection can fail for profiles, poor lighting or unusual poses. The app then uses the full image; check the preview.
- Accuracy is empirical for your chosen dataset and matching thresholds. Do not claim a result until you run evaluation.
