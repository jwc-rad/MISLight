from setuptools import setup, find_namespace_packages

setup(name='mislight',
      packages=find_namespace_packages(include=["mislight", "mislight.*"]),
      version='0.1.0',
      description='MISLight. Medical Imaging in Pytorch Lightning.',
      author='Jae Won Choi',
      license='MIT License',
      install_requires=[
            "torch>=2.0.0",
            "lightning>=2.0.0",
            "monai",
            "SimpleITK",
      ],
      )