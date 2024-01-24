# End-to-end MNIST training demo

This folder contains a demo based on [juharris's project](https://juharris.github.io/train-pytorch-in-js/) that launches a simple model that will train on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in your browser.

A live preview of this demo is available [here](https://carzh.github.io/onnxruntime-training-examples/).

## Run instructions
### 1. Prepare offline artifacts.
On-device-training requires an offline step where 4 training artifacts are generated. The instructions for generating the required artifacts can be found in [`./offline-step/README.md`](offline-step/README.md). 

Once the artifacts have been generated, copy them to [`./web-bundler/public`](web-bundler/public).
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

Once unzipped, they should be copied to the [`web-bundler/public/data`](web-bundler/public/data) folder.

### 3. Run the demo.
```
cd web-bundler
npm install
npm run start
```

## Deploy in Github Pages instructions
The following goes through the steps for deploying to Github Pages in a fork of ONNXRuntime Training Examples repo. 

### 1. Prepare demo
Create a fork of the ONNXRuntime Training Examples repo & follow the above run instructions. Ensure that the local preview is running correctly by checking [http://localhost:9000/](http://localhost:9000/). 

### 2. Deploy branch
Run the following:
```
npm run deploy
```

This will copy all relevant files to a new branch of your fork. By default, files will be copied to a branch called "web-demo."

### 3. Configure Github
Navigate to your repository fork, go to Settings, then Pages. Make sure Source is set to "Deploy from a branch", and then select the "web-demo" branch then "/root" for the folder. Click Save, and your Github Pages site will be built. The same Settings page will direct you to the URL where the live demo is hosted.

![image](https://github.com/carzh/onnxruntime-training-examples/assets/22922935/b35cadce-c961-41b8-8ff2-7ba9922e3e2f)


## onnxruntime-web/training npm package usage
Make sure to use onnxruntime-web version >= 1.17.0, since onnxruntime-web/training is only support in 1.17.0 and above.

The onnxruntime-web package uses WebAssembly binaries, which must be loaded into the browser. The webpack Copy Plugin is the recommended method to load these binaries -- see [`webpack.config.json`](web-bundler/webpack.config.json) for example usage.

If your example is having difficulty recognizing onnxruntime-web/training interfaces, make sure that you have the following settings in your [`tsconfig.json`](web-bundler/tsconfig.json):
```
		"module": "ES2020",
		"moduleResolution": "bundler",
```

## Demo components overview
This section will go over how different components of the demo work together.

This TypeScript demo uses React for its UI, webpack as its bundler, and npm as its package manager. The gh-pages npm package is used to deploy. ORT web for training can be used in both TypeScript and JavaScript projects and can be imported with either a bundler or a script tag. 

The [`webpack.config.js`](./web-bundler/webpack.config.js) file contains bundler configuration information. The webpack bundler will transcompile the files from the [`./src`](./web-bundler/src) directory to a single JavaScript file which will automatically be placed in the [`./public`](./web-bundler/public) directory.

The [`./public`](./web-bundler/public) directory contains the files that will be mounted or served in the browser, which is why we copy the data files and the generated training artifacts to the public folder. A skeleton [`index.html`](./web-bundler/public/index.html) file is required for React projects, which is located in the `./public` directory. For this project, we use an HtmlWebpackPlugin to generate an index.html that correctly imports the generated JavaScript bundle. [`./src/index.tsx`](./web-bundler/src/index.tsx) mounts a React component onto the index root, and [`./src/App.tsx`](./web-bundler/src/App.tsx) contains the code for that React component. Thus, [`./src/App.tsx`](./web-bundler/src/App.tsx) contains the code for creating a TrainingSession, the training loop, and the testing loop, as well as the interface for users.

The ORT web package for training is an npm package, but can be used with a wide array of web frameworks. If using a bundler, ensure that there is a way for the WASM binaries to be mounted or served in the browser. Refer to [`webpack.config.json`](./web-bundler/webpack.config.js) for how this demo uses the CopyPlugin to load the WASM binaries. 

## Demo troubleshooting
If you run into the following errors: `Cannot read properties of undefined (reading 'data')` or `Cannot read properties of undefined (reading 'dims')`, then use Netron to open `training_model.onnx`, double-check the name of the loss output node, and edit [`App.tsx`  line 14](web-bundler/src/App.tsx#L14) to reflect the correct loss output node name.
