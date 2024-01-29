const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = () => {
	return {
		// target: ['web'],
        context: __dirname,
		entry: './src/index.tsx',
		output: {
			path: path.resolve(__dirname, 'public'),
			filename: 'bundle.js',
			library: {
				type: 'umd'
			}
		},
         module: {
            rules: [
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                    },
                }, 
                {
                    test: /\.tsx?$/,
                    use: 'ts-loader',
                    exclude: /node_modules/,
                },
                {
                    test: /\.css$/i,
                    use: ['style-loader', 'css-loader'],
                }]},
		plugins: [
            new CopyPlugin({
			    patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]'}]
		    }), 
            new HtmlWebpackPlugin({
                template: './public/index.html'
        })],
        devServer: {
            hot: true,
            static: {
                directory: path.join(__dirname, 'public'),
            },
            compress: true,
            port: 9000
        },
		mode: 'production',
		resolve: {
			extensions: ['.ts', '.js', '.tsx'],
            fallback: {
                crypto: require.resolve('crypto-browserify'),
                stream: require.resolve("stream-browserify"),
                buffer: require.resolve("buffer/"),
            }
		}
	}
};