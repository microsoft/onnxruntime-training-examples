# Offline step

This step is the same regardless of the method used to import the ONNXRuntime-web/training package. 

## Set-up

Install dependencies. This step requires onnxruntime-training-cpu>=1.17.0. 
```
pip install -r requirements.txt
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu
```

## Generate artifacts

Run the cells in the [Jupyter notebook](./mnist.ipynb).
