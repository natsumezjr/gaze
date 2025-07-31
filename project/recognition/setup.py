from setuptools import setup, find_packages

setup(
    name="gaze-recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mediapipe>=0.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
    ],
    python_requires=">=3.8",
) 