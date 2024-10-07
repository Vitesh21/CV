Feature Detection, Description, and Matching



This notebook, S20220010126_FDDM.ipynb, performs feature detection, description, and matching on two images using Harris Corner Detection and SIFT (Scale-Invariant Feature Transform).



Requirements
Make sure you have the following libraries installed:

OpenCV
NumPy
Matplotlib
Install them with:

bash
Copy code
pip install opencv-python-headless numpy matplotlib.


Code Overview
This notebook is divided into three main parts:

Harris Corner Detection: Detects corner features in each image.
SIFT Keypoint Detection: Detects and describes keypoints using SIFT.
Feature Matching: Matches the features of the two images based on SIFT descriptors using a ratio test.



Outputs
The notebook displays and saves the following results as images:

Harris Corner Detection (saved as S20220010126_FDDM_output1.png)
SIFT Keypoints (saved as S20220010126_FDDM_output2.png)
Feature Matching (saved as S20220010126_FDDM_output3.png)




Usage
Open the notebook S20220010126_FDDM.ipynb in Jupyter Notebook or JupyterLab.
Run the cells in sequence to see the results.
The processed images are saved in the working directory.

