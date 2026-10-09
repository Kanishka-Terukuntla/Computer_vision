# Computer Vision Lab Activities

A collection of Computer Vision lab assignments, implementations, experiments, and results completed as part of academic coursework.

The repository covers fundamental image processing techniques, frequency-domain analysis, image segmentation, watershed segmentation, and stereo vision.

## 📂 Repository Structure

```text
Computer_vision/
├── lab1/       # Image filtering and smoothing
├── lab2/       # Frequency-domain image processing
├── lab3/       # Computer vision lab experiments
├── lab4/       # Lab activities
├── lab6/       # Watershed segmentation and coin counting
├── lab7/       # Stereo vision and depth estimation
├── .gitignore
└── requirements.txt
```

> **Note:** The `notes/` directory is excluded from Git tracking and is not included in the repository.

## 🧪 Lab Activities

### Lab 1 — Image Filtering

Explores spatial-domain filtering techniques for image smoothing and noise reduction.

- Mean filtering
- Median filtering
- Sequential application of mean and median filters
- Comparison of built-in and manually implemented filters
- Visualization of filtered outputs

### Lab 2 — Frequency-Domain Image Processing

Explores image representation and processing in the frequency domain.

- Fourier transform concepts
- Magnitude and phase information
- Combining the magnitude spectrum of one image with the phase spectrum of another
- Reconstruction and visualization of processed images

### Lab 3 — Image Processing Experiments

Contains lab experiments involving image processing, with supporting images and Jupyter Notebook implementations.

### Lab 4 — Lab Activities

Reserved for additional Computer Vision lab exercises and implementations.

### Lab 6 — Image Segmentation

Explores image segmentation techniques, including watershed-based segmentation.

- Watershed segmentation
- Marker-based segmentation
- Connected-component analysis
- Counting coins and circular objects
- Handling overlapping objects
- Visualization of segmentation results

### Lab 7 — Stereo Vision

Explores stereo vision techniques to estimate scene depth from a pair of images.

- Stereo image pairs
- Disparity map computation
- Depth estimation
- Comparison with ground-truth disparity
- Error-map visualization and quantitative evaluation

## 🛠️ Technologies Used

- **Programming language:** Python
- **Libraries:** OpenCV, NumPy, Matplotlib, and other dependencies listed in `requirements.txt`
- **Environment:** Jupyter Notebook
- **Version control:** Git and GitHub

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Kanishka-Terukuntla/Computer_vision.git
cd Computer_vision
```

### 2. Create a virtual environment

```bash
python -m venv opencvenv
```

Activate it on Windows PowerShell:

```powershell
.\opencvenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

If Jupyter is installed in your environment:

```bash
jupyter notebook
```

Open the relevant lab folder and run the notebook cells in order.

## 📊 Results and Experiments

The repository includes notebooks, sample images, and selected generated outputs used to implement and evaluate Computer Vision techniques.

Results may vary depending on input images, algorithm parameters, and execution environments.

## 🎯 Learning Objectives

- Understand spatial and frequency-domain image processing.
- Implement and compare image filtering techniques.
- Explore image segmentation and morphological operations.
- Apply watershed segmentation to separate overlapping objects.
- Understand disparity maps and stereo-based depth estimation.
- Develop practical skills in Python and OpenCV.

## 📌 Notes

- Run each notebook from its respective lab directory when relative file paths are used.
- Ensure that the required input images are available before executing a notebook.
- Generated images and intermediate results are included where relevant.
- The `notes/` folder is intentionally excluded from version control.

## 👩‍💻 Author

**Kanishka Terukuntla**

B.Tech — Computer Science and Engineering (AI)

[GitHub Profile](https://github.com/Kanishka-Terukuntla)

---

*Academic coursework and practical implementations in Computer Vision.*