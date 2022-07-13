from setuptools import setup, find_namespace_packages

setup(name='mislight',
      packages=find_namespace_packages(include=["mislight", "mislight.*"]),
      version='0.1.0',
      description='MISLight. Medical Image Segmentation in Pytorch Lightning.',
      author='Jae Won Choi',
      license='MIT License',
      install_requires=[
            "torch>=1.10.0",
            "pytorch-lightning",
            "monai",
            "scikit-image",
            "SimpleITK==2.0.2",
            "parmap",
            "connected-components-3d",
            "fastremap",
            
            # FLARE22 evaluation
            "loguru",
            "pynvml",
            "psutil",
      ],
      )
