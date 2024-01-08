import { Button, Container, Grid, Link, TextField, Switch, FormControlLabel} from '@mui/material';
import React from 'react';
import './App.css';
import Plot from 'react-plotly.js';
import * as ort from 'onnxruntime-web/training';
import { MnistData } from './mnist';
import { Digit } from './Digit';

// TODO: change the copy
function App() {
    const numRows = 28;
    const numCols = 28;

    const logIntervalMs = 1000;
    const waitAfterLoggingMs = 500;

	// const [maxNumTrainSamples, setMaxNumTrainSamples] = React.useState<number>(MnistData.MAX_NUM_TRAIN_SAMPLES);
	// const [maxNumTestSamples, setMaxNumTestSamples] = React.useState<number>(MnistData.MAX_NUM_TEST_SAMPLES);

	const [maxNumTrainSamples, setMaxNumTrainSamples] = React.useState<number>(6400);
	const [maxNumTestSamples, setMaxNumTestSamples] = React.useState<number>(1280);

	const [batchSize, setBatchSize] = React.useState<number>(MnistData.BATCH_SIZE);
    // TODO: change default number of epochs back to 5
	const [numEpochs, setNumEpochs] = React.useState<number>(2);

    const [trainingLosses, setTrainingLosses] = React.useState<number[]>([]);
	const [testAccuracies, setTestAccuracies] = React.useState<number[]>([]);

    const [digits, setDigits] = React.useState<{ pixels: Float32Array, label: number }[]>([])
	const [digitPredictions, setDigitPredictions] = React.useState<number[]>([])

    const [enableLiveLogging, setEnableLiveLogging] = React.useState<boolean>(false);

    const [statusMessage, setStatusMessage] = React.useState("");
    const [errorMessage, setErrorMessage] = React.useState("");
	const [messages, setMessages] = React.useState<string[]>([]);

    let messagesQueue = [];

    let lastLogTime = 0;

    function showStatusMessage(message: string) {
        console.log(message);
        setStatusMessage(message);
    }

    function showErrorMessage(message: string) {
        console.log(message);
        setErrorMessage(message);
    }

    function addMessages(messagesToAdd: string[]) {
        setMessages(messages => [...messages, ...messagesToAdd]);
    }

    function addMessageToQueue(message: string) {
        messagesQueue.push(message);
    }

    function clearOutputs() {
        setTrainingLosses([]);
        setTestAccuracies([]);
        setMessages([]);
        setStatusMessage("");
        setErrorMessage("");
        messagesQueue = [];
    }

    async function logMessage(message: string) {
        addMessageToQueue(message);
        if (Date.now() - lastLogTime > logIntervalMs) {
            showStatusMessage(message);
            if (enableLiveLogging) {
                addMessages(messagesQueue);
                messagesQueue = [];
            }
            // wait for UI to update before updating lastLogTime, otherwise will have lags in updates
            await new Promise(r => setTimeout(r, waitAfterLoggingMs));
            lastLogTime = Date.now();
        }
    }

    function indexOfMax(arr: Float32Array): number {
        if (arr.length === 0) {
            throw new Error('index of max (used in test accuracy function) expects a non-empty array. Something went wrong.');
        }

        let maxIndex = 0;
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    function getPredictions(results: ort.Tensor): number[] {
        const predictions = [];
        const [batchSize, numClasses] = results.dims;
        for (let i = 0; i < batchSize; ++i) {  
            const probabilities = results.data.slice(i * numClasses, (i + 1) * numClasses) as Float32Array;
            const resultsLabel = indexOfMax(probabilities);
            predictions.push(resultsLabel);
        }
        return predictions;
    }

    function countCorrectPredictions(output: ort.Tensor, labels: ort.Tensor): number {
        let result = 0;
        const predictions = getPredictions(output);
        for (let i = 0; i < predictions.length; ++i) {
            if (BigInt(predictions[i]) === labels.data[i]) {
                ++result;
            }
        }
        return result;
    }

    function getPixels(data: Float32Array, numRows: number, numCols: number) {
		const result: number[][] = []
		for (let row = 0; row < numRows; ++row) {
			const rowPixels: number[] = []
			for (let col = 0; col < numCols; ++col) {
				rowPixels.push(data[row * numCols + col])
			}
			result.push(rowPixels)
		}
		return result
	}

    async function loadTrainingSession(): Promise<ort.TrainingSession> {
        showStatusMessage('Attempting to load training session...');

        const chkptPath = 'checkpoint';
        const trainingPath = 'training_model.onnx';
        const optimizerPath = 'optimizer_model.onnx';
        const evalPath = 'eval_model.onnx';

        const createOptions: ort.TrainingSessionCreateOptions = {
                    checkpointState: chkptPath, 
                    trainModel: trainingPath, 
                    evalModel: evalPath, 
                    optimizerModel: optimizerPath};

        try {
            const session = await ort.TrainingSession.create(createOptions);
            showStatusMessage('Training session loaded');
            return session;
        } catch (err) {
            showErrorMessage('Error loading the training session: ' + err);
            console.log("Error loading the training session: " + err);
            throw err;
        }
    };

    async function updateDigitPredictions(session: ort.TrainingSession) {
        const input = new Float32Array(digits.length * numRows * numCols);
        const batchShape = [digits.length, numRows * numCols];
        const labels = [];
        for (let i = 0; i < digits.length; ++i) {
            const pixels = digits[i].pixels;
            for (let j = 0; j < pixels.length; ++j) {
                input[i * pixels.length + j] = MnistData.normalize(pixels[j]);
            }
           labels.push(BigInt(digits[i].label)); 
        }

        const feeds = {
            input: new ort.Tensor('float32', input, batchShape),
            labels: new ort.Tensor('int64', new BigInt64Array(labels), [digits.length])
        };

        const results = await session.runEvalStep(feeds);
        const predictions = getPredictions(results['output']);
        setDigitPredictions(predictions.slice(0, digits.length));
    }

    async function runTrainingEpoch(session: ort.TrainingSession, dataSet: MnistData, epoch: number) {
        let batchNum =  0;
        let totalNumBatches = dataSet.getNumTrainingBatches();
        for await (const batch of dataSet.trainingBatches()) {
            ++batchNum;
            const feeds = {
                input: batch.data,
                labels: batch.labels
            }

            const results = await session.runTrainStep(feeds);

            // updating UI
            const loss = parseFloat(results['onnx::loss::2'].data);
            setTrainingLosses(losses => losses.concat(loss));
            const message = `TRAINING | Epoch: ${String(epoch + 1).padStart(2)} / ${numEpochs} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Loss: ${loss.toFixed(4)}`;
            await logMessage(message);

            await session.runOptimizerStep();
            await session.lazyResetGrad();
            await updateDigitPredictions(session);
        }
    }

    async function runTestingEpoch(session: ort.TrainingSession, dataSet: MnistData, epoch: number): Promise<number> {
        let batchNum = 0;
        let totalNumBatches = dataSet.getNumTestBatches();
        let numCorrect = 0;
        let testPicsSoFar = 0;
        let accumulatedLoss = 0;
        for await (const batch of dataSet.testBatches()) {
            ++batchNum;
            const feeds = {
                input: batch.data,
                labels: batch.labels
            }

            const results = await session.runEvalStep(feeds);

            // update UI
            const loss = parseFloat(results['onnx::loss::2'].data);
            accumulatedLoss += loss;
            testPicsSoFar += batch.data.dims[0];
            numCorrect += countCorrectPredictions(results['output'], batch.labels);
            const message = `TESTING | Epoch: ${String(epoch + 1).padStart(2)} / ${numEpochs} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Average test loss: ${(accumulatedLoss / batchNum).toFixed(4)} | Accuracy: ${numCorrect}/${testPicsSoFar} (${(100 * numCorrect / testPicsSoFar).toFixed(2)}%)`;
            await logMessage(message);
        }
        const avgAcc = numCorrect / testPicsSoFar;
        setTestAccuracies(accs => accs.concat(avgAcc));
        return avgAcc;
    }

    async function train() {
        clearOutputs();
        const trainingSession = await loadTrainingSession();
        const dataSet = new MnistData(batchSize, maxNumTrainSamples, maxNumTestSamples);

        lastLogTime = Date.now();
        await updateDigitPredictions(trainingSession);
        showStatusMessage('Training started');
        let testAcc = 0;
        for (let epoch = 0; epoch < numEpochs; epoch++) {
            await runTrainingEpoch(trainingSession, dataSet, epoch);
            testAcc = await runTestingEpoch(trainingSession, dataSet, epoch);
        }
        showStatusMessage(`Training completed. Final test set accuracy: ${(100 * testAcc).toFixed(2)}%`);
    }

    function renderPlots() {
		const margin = { t: 20, r: 25, b: 25, l: 40 }
		return (<div className="section">
			<h3>Plots</h3>
			<Grid container spacing={2}>
				<Grid item xs={12} md={6}>
					<h4>Training Loss</h4>
					<Plot
						data={[
							{
								x: trainingLosses.map((_, i) => i),
								y: trainingLosses,
								type: 'scatter',
								mode: 'lines',
							}
						]}
						layout={{ margin, width: 400, height: 320 }}
					/>
				</Grid><Grid item xs={12} md={6}>
					<h4>Test Accuracy (%)</h4>
					<Plot
						data={[
							{
								x: testAccuracies.map((_, i) => i + 1),
								y: testAccuracies.map(a => 100 * a),
								type: 'scatter',
								mode: 'lines+markers',
							}
						]}
						layout={{ margin, width: 400, height: 320 }}
					/>
				</Grid>
			</Grid>
		</div>)
	}
    
    function renderDigits() {
		return (<div className="section">
			<h4>Test Digits</h4>
			<Grid container spacing={2}>
				{digits.map((digit, digitIndex) => {
					const { pixels, label } = digit
					const rgdPixels = getPixels(pixels, numRows, numCols)
					return (<Grid key={digitIndex} item xs={6} sm={3} md={2}>
						<Digit pixels={rgdPixels} label={label} prediction={digitPredictions[digitIndex]} />
					</Grid>)
				})}
			</Grid>
		</div>)
	}
    
    const loadDigits = React.useCallback(async () => {
		const maxNumDigits = 18
		const seenLabels = new Set()
		const dataSet = new MnistData()
		dataSet.maxNumTestSamples = 2 * dataSet.batchSize
		const digits = []
		const normalize = false
		for await (const testBatch of dataSet.testBatches(normalize)) {
			const { data, labels } = testBatch
			const batchSize = labels.dims[0]
			const size = data.dims[1]
			for (let i = 0; digits.length < maxNumDigits && i < batchSize; ++i) {
				const label = Number(labels.data[i])
				if (seenLabels.size < 10 && seenLabels.has(label)) {
					continue
				}
				seenLabels.add(label)
				const pixels = data.data.slice(i * size, (i + 1) * size) as Float32Array

				digits.push({ pixels, label })
			}

			if (digits.length >= maxNumDigits) {
				break
			}
		}
		setDigits(digits)
	}, [])

	React.useEffect(() => {
		loadDigits()
	}, [loadDigits])

        return (
            <Container className = "App">
            <div className="section">
			<h2>ONNX Runtime Web Training Demo</h2>
			<p>
				In this example, you'll a train classifier in your browser to recognize handwritten digits from the <Link href="https://deepai.org/dataset/mnist" target="_blank" rel="noopener">MNIST Dataset</Link>.
			</p>
			<p>
				You can learn more about how to set up a model that can be trained in your browser at <Link href="https://github.com/juharris/train-pytorch-in-js" target="_blank" rel="noopener">github.com/juharris/train-pytorch-in-js</Link>.
			</p>
		</div>
		<div className="section">
			<h3>Training</h3>
			<p>
				After each epoch, the learning rate will be multiplied by <code>Gamma</code>.
			</p>
		</div>
		<div className="section">
			<Grid container spacing={{ xs: 1, md: 2 }}>
				<Grid item xs={12} md={4} >
					<TextField label="Number of epochs"
						type="number"
						value={numEpochs}
						onChange={(e) => setNumEpochs(Number(e.target.value))}
					/>
				</Grid>
				<Grid item xs={12} md={4}>
					<TextField label="Batch size"
						type="number"
						value={batchSize}
						onChange={(e) => setBatchSize(Number(e.target.value))}
					/>
				</Grid>
			</Grid>
		</div>
		<div className="section">
			<Grid container spacing={{ xs: 1, md: 2 }}>
				<Grid item xs={12} md={4} >
					<TextField type="number"
						label="Max number of training samples"
						value={maxNumTrainSamples}
						onChange={(e) => setMaxNumTrainSamples(Number(e.target.value))}
					/>
				</Grid>
				<Grid item xs={12} md={4}>
					<TextField type="number"
						label="Max number of test samples"
						value={maxNumTestSamples}
						onChange={(e) => setMaxNumTestSamples(Number(e.target.value))}
					/>
				</Grid>
			</Grid>

        </div>
        <div className="section">
            <FormControlLabel 
                control={<Switch 
                    checked={enableLiveLogging}
                    onChange={(e) => setEnableLiveLogging(!enableLiveLogging)} />} 
                label='Log all batch results as they happen. Can slow down training.' />
        </div>
        <div className="section">
            <Button onClick={train}
                variant='contained'>
                Train
            </Button>
        </div>
        <pre>{statusMessage}</pre>

        {renderPlots()}

        {renderDigits()}

		{messages.length > 0 &&
			<div>
				<h3>Logs:</h3>
				<pre>
					{messages.map((m, i) => (<React.Fragment key={i}>
						{m}
						<br />
					</React.Fragment>))}
				</pre>
			</div>}
		{errorMessage &&
			<p className='error'>
				{errorMessage}
			</p>}
	</Container> 
        );
}

export default App;