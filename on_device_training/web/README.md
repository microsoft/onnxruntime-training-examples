# End-to-end MNIST training demo

This folder contains a demo based on [juharris's project](https://juharris.github.io/train-pytorch-in-js/) that launches a simple model that will train on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in your browser.

## Run instructions
### 1. Prepare offline artifacts.
On-device-training requires an offline step where 4 training artifacts are generated. The instructions for generating the required artifacts can be found in `./offline-step/README.md`. 

Once the artifacts have been generated, copy them to `./web-bundler/public`
```
cp offline-step/*.onnx web-bundler/public
cp offline-step/checkpoint web-bundler/public
```

### 2. Prepare data.
This demo uses the MNIST dataset, which can be downloaded from [here](http://yann.lecun.com/exdb/mnist/). 

Download the following files:
* train-labels-idx1-ubyte.gz
* train-images-idx3-ubyte.gz
* t10k-labels-idx1-ubyte.gz
* t10k-images-idx3-ubyte.gz

You can use the gzip utility to unzip them.
```
gunzip train-labels-idx1-ubyte.gz train-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz
```

Once unzipped, they should be copied to the `web-bundler/public/data` folder.

### 3. Run the demo.
```
cd web-bundler
npm install
npm run start
```

## onnxruntime-web/training npm package usage
The onnxruntime-web package uses WebAssembly binaries, which must be loaded into the browser. The webpack Copy Plugin is recommended.

If your example is having difficulty recognizing onnxruntime-web/training interfaces, make sure that you have the following settings in your tsconfig.json:
```
		"module": "ES2020",
		"moduleResolution": "bundler",
```

Also make sure that onnxruntime-web version >= 1.17.0, since onnxruntime-web/training is only support in 1.17.0 and above.

## Demo troubleshooting
If you run into the following errors: `Cannot read properties of undefined (reading 'data')` or `Cannot read properties of undefined (reading 'dims')`, then use Netron to open `training_model.onnx`, double-check the name of the loss output node, and edit `App.tsx` line 13 to reflect the correct loss output node name.

