// Modified from https://github.com/cazala/mnist/blob/master/src/mnist.js
// so that we can place the data in a specific folder and avoid out of memory errors
// and use TypeScript.
// TODO: modify comments to credit justin
import * as ort from 'onnxruntime-web/training';

// Assume the data was loaded when running the Python scripts.
/**
 * Dataset description at https://deepai.org/dataset/mnist.
 */
export class MnistData {
	static readonly BATCH_SIZE = 64
	static readonly MAX_NUM_TRAIN_SAMPLES = 60000
	static readonly MAX_NUM_TEST_SAMPLES = 10000

	static readonly pixelMax = 255
	static readonly pixelMean = 0.1307
	static readonly pixelStd = 0.3081

	constructor(
		public batchSize = MnistData.BATCH_SIZE,
		public maxNumTrainSamples = MnistData.MAX_NUM_TRAIN_SAMPLES,
		public maxNumTestSamples = MnistData.MAX_NUM_TEST_SAMPLES,
	) {
		if (batchSize <= 0) {
			throw new Error("batchSize must be > 0")
		}
	}

	public getNumTrainingBatches(): number {
		return Math.floor(this.maxNumTrainSamples / this.batchSize)
	}

	public getNumTestBatches(): number {
		return Math.floor(this.maxNumTestSamples / this.batchSize)
	}

	private *batches(data: ort.Tensor[], labels: ort.Tensor[]) {
		for (let batchIndex = 0; batchIndex < data.length; ++batchIndex) {
			yield {
				data: data[batchIndex],
				labels: labels[batchIndex],
			}
		}
	}

	public async * trainingBatches() {
		// Avoid keeping data in memory.
		const trainingData = await this.getData('data/train-images-idx3-ubyte', 2051, 'data', this.maxNumTrainSamples)
		const trainingLabels = await this.getData('data/train-labels-idx1-ubyte', 2049, 'labels', this.maxNumTrainSamples)
		yield* this.batches(trainingData, trainingLabels)
	}

	public async * testBatches(normalize = true) {
		// Avoid keeping data in memory.
		const testData = await this.getData('data/t10k-images-idx3-ubyte', 2051, 'data', this.maxNumTestSamples, normalize)
		const testLabels = await this.getData('data/t10k-labels-idx1-ubyte', 2049, 'labels', this.maxNumTestSamples)
		yield* this.batches(testData, testLabels)
	}

	private async getData(url: string, expectedMagicNumber: number, dataType: 'data' | 'labels', maxNumSamples: number, normalize = true): Promise<ort.Tensor[]> {
		console.debug(`Loading ${dataType} from "${url}".`)
		const result = []
		const response = await fetch(url)
		const buffer = await response.arrayBuffer()
		if (buffer.byteLength < 16) {
			throw new Error("Invalid MNIST images file. There aren't enough bytes")
		}
		const magicNumber = new DataView(buffer.slice(0, 4)).getInt32(0, false)
		if (magicNumber !== expectedMagicNumber) {
			throw new Error(`Invalid MNIST images file. The magic number is not ${expectedMagicNumber}. Got ${magicNumber}.`)
		}
		const numDimensions = new DataView(buffer.slice(3, 4)).getUint8(0)
		const shape = []
		for (let i = 0; i < numDimensions; ++i) {
			shape.push(new DataView(buffer.slice(4 + i * 4, 8 + i * 4)).getUint32(0, false))
		}
		const numItems = shape[0]
		const dimensions = shape.slice(1)
		const dataSize = dimensions.reduce((a, b) => a * b, 1);
		const batchShape: number[] = dataType === 'data' ? [this.batchSize, dataSize] : [this.batchSize]

		let offset = 4 + 4 * shape.length
		for (let i = 0; i < numItems; i += this.batchSize) {
			if (i + this.batchSize > maxNumSamples) {
				break
			}

			if (buffer.byteLength < offset + this.batchSize * dataSize) {
				break
			}
			let batch
			switch (dataType) {
				case 'data':
					const images = new Uint8Array(buffer.slice(offset, offset + this.batchSize * dataSize))
					batch = (new Float32Array(images))
					if (normalize) {
						batch = batch.map(v => MnistData.normalize(v))
					}
					batch = new ort.Tensor('float32', batch, batchShape)
					break
				case 'labels':
					const labels = new Uint8Array(buffer.slice(offset, offset + this.batchSize * dataSize))
					batch = Array.from(labels).map(BigInt)
					batch = new ort.Tensor('int64', new BigInt64Array(batch), batchShape)
					break
			}

			result.push(batch)
			offset += this.batchSize * dataSize
		}

		return result
	}

	public static normalize(pixelValue: number): number {
		return ((pixelValue / this.pixelMax) - this.pixelMean) / this.pixelStd
	}
}