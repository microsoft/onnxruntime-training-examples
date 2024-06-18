import { Button, Container, Grid, Link, TextField, Switch, FormControlLabel, Table, TableHead, TableBody, TableRow, TableCell, TableContainer } from '@mui/material';
import React from 'react';
import './App.css';
import Plot from 'react-plotly.js';
import * as ort from 'onnxruntime-web/training';
import { MnistData } from './mnist';
import { Digit } from './Digit';

function App() {
    // constants
    const numRows = 28;
    const numCols = 28;

    const lossNodeName = "onnx::loss::8";

    const logIntervalMs = 1000;
    const waitAfterLoggingMs = 500;
    let lastLogTime = 0;

    let messagesQueue = [];

    // React components
    const [maxNumTrainSamples, setMaxNumTrainSamples] = React.useState<number>(6400);
    const [maxNumTestSamples, setMaxNumTestSamples] = React.useState<number>(1280);

    const [batchSize, setBatchSize] = React.useState<number>(MnistData.BATCH_SIZE);
    const [numEpochs, setNumEpochs] = React.useState<number>(5);

    const [trainingLosses, setTrainingLosses] = React.useState<number[]>([]);
    const [testAccuracies, setTestAccuracies] = React.useState<number[]>([]);

    const [digits, setDigits] = React.useState<{ pixels: Float32Array, label: number }[]>([])
    const [digitPredictions, setDigitPredictions] = React.useState<number[]>([])

    const [isTraining, setIsTraining] = React.useState<boolean>(false);

    const [moreInfoIsCollapsed, setMoreInfoIsCollapsed] = React.useState<boolean>(true);

    // logging React components
    const [enableLiveLogging, setEnableLiveLogging] = React.useState<boolean>(false);

    const [statusMessage, setStatusMessage] = React.useState("");
    const [errorMessage, setErrorMessage] = React.useState("");
    const [messages, setMessages] = React.useState<string[]>([]);

    // logging helper functions

    function toggleMoreInfoIsCollapsed() {
        setMoreInfoIsCollapsed(!moreInfoIsCollapsed);
    }

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

    // training & testing helper functions
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
            optimizerModel: optimizerPath
        };

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

    // training & testing functions
    async function runTrainingEpoch(session: ort.TrainingSession, dataSet: MnistData, epoch: number) {
        let batchNum = 0;
        let totalNumBatches = dataSet.getNumTrainingBatches();
        const epochStartTime = Date.now();
        let iterationsPerSecond = 0;
        await logMessage(`TRAINING | Epoch: ${String(epoch + 1).padStart(2)} / ${numEpochs} | Starting training...`)
        for await (const batch of dataSet.trainingBatches()) {
            ++batchNum;
            // create input
            const feeds = {
                input: batch.data,
                labels: batch.labels
            }

            // call train step
            const results = await session.runTrainStep(feeds);

            // updating UI with metrics
            const loss = parseFloat(results[lossNodeName].data);
            setTrainingLosses(losses => losses.concat(loss));
            iterationsPerSecond = batchNum / ((Date.now() - epochStartTime) / 1000);
            const message = `TRAINING | Epoch: ${String(epoch + 1).padStart(2)} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Loss: ${loss.toFixed(4)} | ${iterationsPerSecond.toFixed(2)} it/s`;
            await logMessage(message);

            // update weights then reset gradients
            await session.runOptimizerStep();
            await session.lazyResetGrad();
            // update digit predictions
            await updateDigitPredictions(session);
        }
        return iterationsPerSecond;
    }

    async function runTestingEpoch(session: ort.TrainingSession, dataSet: MnistData, epoch: number): Promise<number> {
        let batchNum = 0;
        let totalNumBatches = dataSet.getNumTestBatches();
        let numCorrect = 0;
        let testPicsSoFar = 0;
        let accumulatedLoss = 0;
        const epochStartTime = Date.now();
        await logMessage(`TESTING | Epoch: ${String(epoch + 1).padStart(2)} / ${numEpochs} | Starting testing...`)
        for await (const batch of dataSet.testBatches()) {
            ++batchNum;

            // create input
            const feeds = {
                input: batch.data,
                labels: batch.labels
            }

            // call eval step
            const results = await session.runEvalStep(feeds);

            // update UI with metrics
            const loss = parseFloat(results[lossNodeName].data);
            accumulatedLoss += loss;
            testPicsSoFar += batch.data.dims[0];
            numCorrect += countCorrectPredictions(results['output'], batch.labels);
            const iterationsPerSecond = batchNum / ((Date.now() - epochStartTime) / 1000);
            const message = `TESTING | Epoch: ${String(epoch + 1).padStart(2)} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Average test loss: ${(accumulatedLoss / batchNum).toFixed(2)} | Accuracy: ${numCorrect}/${testPicsSoFar} (${(100 * numCorrect / testPicsSoFar).toFixed(2)}%) | ${iterationsPerSecond.toFixed(2)} it/s`;
            await logMessage(message);
        }
        const avgAcc = numCorrect / testPicsSoFar;
        setTestAccuracies(accs => accs.concat(avgAcc));
        return avgAcc;
    }

    async function train() {
        clearOutputs();

        setIsTraining(true);
        if (maxNumTrainSamples > MnistData.MAX_NUM_TRAIN_SAMPLES || maxNumTestSamples > MnistData.MAX_NUM_TEST_SAMPLES) {
            showErrorMessage(`Max number of training samples (${maxNumTrainSamples}) or test samples (${maxNumTestSamples}) exceeds the maximum allowed (${MnistData.MAX_NUM_TRAIN_SAMPLES} and ${MnistData.MAX_NUM_TEST_SAMPLES}, respectively). Please try again.`);
            return;
        }

        const trainingSession = await loadTrainingSession();
        const dataSet = new MnistData(batchSize, maxNumTrainSamples, maxNumTestSamples);

        lastLogTime = Date.now();
        await updateDigitPredictions(trainingSession);
        const startTrainingTime = Date.now();
        showStatusMessage('Training started');
        let itersPerSecCumulative = 0;
        let testAcc = 0;
        for (let epoch = 0; epoch < numEpochs; epoch++) {
            itersPerSecCumulative += await runTrainingEpoch(trainingSession, dataSet, epoch);
            testAcc = await runTestingEpoch(trainingSession, dataSet, epoch);
        }
        const trainingTimeMs = Date.now() - startTrainingTime;
        showStatusMessage(`Training completed. Final test set accuracy: ${(100 * testAcc).toFixed(2)}% | Total training time: ${trainingTimeMs / 1000} seconds | Average iterations / second: ${(itersPerSecCumulative / numEpochs).toFixed(2)}`);
        setIsTraining(false);
    }

    // plots and digits
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

    // component HTML
    return (
        <Container className="App">
            <div className="section">
                <h2>ONNX Runtime Web Training Demo</h2>
                <p>
                    This demo showcases using <Link href="https://onnxruntime.ai/docs/">ONNX Runtime Training for Web</Link> to train a simple neural network that recognizes handwritten digits from the <Link href='http://yann.lecun.com/exdb/mnist/'>MNIST dataset</Link>.
                    The MNIST dataset consists of 28x28 grayscale images of handwritten digits and labels for each image.
                    The training set contains {MnistData.MAX_NUM_TRAIN_SAMPLES} images and the test set contains {MnistData.MAX_NUM_TEST_SAMPLES} images.
                    The training artifacts for the model and its weights altogether take up 1.59MB of storage, with the largest components being the 1.59MB checkpoint file and the 4KB training model ONNX file.
                </p>
            </div>
            <div className="section">
                <h3>Background</h3>
                <p>
                    ONNX Runtime Training for Web is a new feature in ORT 1.17.0 that enables developers to train machine learning models in the browser using CPU and WebAssembly.
                </p>
                <p>
                    This in-browser training capability is specifically designed to support federated learning scenarios, where multiple devices can collaborate to train a model without sharing data with each other.
                    This approach enhances privacy and security while still allowing for effective machine learning.
                </p>
                <p>
                    If you're interested in learning more about ONNX Runtime Training for Web and its potential applications, be sure to check out our blog coming out soon.
                </p>
                <p>
                    For more information on how to use ONNX Runtime Web for training, please refer to <Link href="https://onnxruntime.ai/docs/">ONNX Runtime documentation</Link> or
                    the <Link href="https://github.com/microsoft/onnxruntime-training-examples">ONNX Runtime Training Examples code</Link>.
                </p>
            </div>
            <div className="section">
                <h3>Training Metrics</h3>
                <TableContainer sx={{ width: '50%' }}>
                    <Table size='small'>
                        <TableHead>
                            <TableRow>
                                <TableCell sx={{ fontWeight: 'bold' }}>Browser</TableCell>
                                <TableCell sx={{ fontWeight: 'bold' }}>Heap usage in MB</TableCell>
                                <TableCell sx={{ fontWeight: 'bold' }}>it/s</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                <TableCell>Chrome</TableCell>
                                <TableCell>25.2</TableCell>
                                <TableCell>54.30</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>Edge</TableCell>
                                <TableCell>24.2</TableCell>
                                <TableCell>55.48</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
                <div className="section moreInfo">
                    <Button onClick={toggleMoreInfoIsCollapsed}>{moreInfoIsCollapsed ? 'Expand' : 'Collapse'} more info</Button>
                    {!moreInfoIsCollapsed && <div>
                        <p>
                            The above measurements were obtained on a Windows PC in a window with a single tab open.
                        </p>
                        <p>
                            Measuring memory usage and performance in the browser is difficult because things such as screen resolution, window size, OS and OS version of the host machine, the number of tabs or windows open, the number of extensions installed, and more can affect memory usage.
                            Thus, the above results may be difficult to replicate. The above numbers are meant to reflect that training in the browser does not have to be compute- or memory-intensive when using the ORT Web for Training framework.
                        </p>
                    </div>}
                </div>
            </div>
            <div className="section">
                <h3>Training</h3>
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
                    disabled={isTraining}
                    variant='contained'>
                    Train
                </Button>
                <br></br>
            </div>
            <pre>{statusMessage}</pre>
            {errorMessage &&
                <p className='error'>
                    {errorMessage}
                </p>}

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
        </Container>
    );
}

export default App;