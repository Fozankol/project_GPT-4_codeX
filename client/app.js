const elements = {
  fileInput: document.querySelector('#fileInput'),
  trainingText: document.querySelector('#trainingText'),
  textStats: document.querySelector('#textStats'),
  modelStatus: document.querySelector('#modelStatus'),
  epochsInput: document.querySelector('#epochsInput'),
  contextInput: document.querySelector('#contextInput'),
  embeddingInput: document.querySelector('#embeddingInput'),
  hiddenInput: document.querySelector('#hiddenInput'),
  learningRateInput: document.querySelector('#learningRateInput'),
  samplesInput: document.querySelector('#samplesInput'),
  trainButton: document.querySelector('#trainButton'),
  stopButton: document.querySelector('#stopButton'),
  progressFill: document.querySelector('#progressFill'),
  progressText: document.querySelector('#progressText'),
  lossLog: document.querySelector('#lossLog'),
  seedInput: document.querySelector('#seedInput'),
  lengthInput: document.querySelector('#lengthInput'),
  temperatureInput: document.querySelector('#temperatureInput'),
  generateButton: document.querySelector('#generateButton'),
  outputText: document.querySelector('#outputText'),
};

const exampleText = `We train this tiny AI from scratch.
It starts with random numbers and learns only from the text you provide.
Give it a larger file, more epochs, and a clear writing style to get better samples.`;

const state = {
  model: null,
  stopRequested: false,
};

class ScratchTextModel {
  constructor(text, config) {
    this.text = text;
    this.config = config;
    this.vocab = [...new Set(text)].sort();
    this.charToIndex = new Map(this.vocab.map((char, index) => [char, index]));
    this.indices = Uint16Array.from(text, char => this.charToIndex.get(char));
    this.vocabSize = this.vocab.length;
    this.inputSize = config.contextSize * config.embeddingSize;

    this.embeddings = this.createRandomArray(
      this.vocabSize * config.embeddingSize,
      0.08
    );
    this.hiddenWeights = this.createRandomArray(
      this.inputSize * config.hiddenSize,
      Math.sqrt(2 / this.inputSize)
    );
    this.hiddenBias = new Float32Array(config.hiddenSize);
    this.outputWeights = this.createRandomArray(
      config.hiddenSize * this.vocabSize,
      Math.sqrt(2 / config.hiddenSize)
    );
    this.outputBias = new Float32Array(this.vocabSize);
  }

  createRandomArray(size, scale) {
    const values = new Float32Array(size);

    for (let index = 0; index < size; index++) {
      values[index] = (Math.random() * 2 - 1) * scale;
    }

    return values;
  }

  getSamplePositions(maxSamples) {
    const positions = [];
    const total = this.indices.length - this.config.contextSize;
    const step = Math.max(1, Math.floor(total / maxSamples));

    for (
      let index = this.config.contextSize;
      index < this.indices.length && positions.length < maxSamples;
      index += step
    ) {
      positions.push(index);
    }

    return positions;
  }

  shuffle(values) {
    for (let index = values.length - 1; index > 0; index--) {
      const swapIndex = Math.floor(Math.random() * (index + 1));
      const value = values[index];
      values[index] = values[swapIndex];
      values[swapIndex] = value;
    }
  }

  readContext(position) {
    const context = new Uint16Array(this.config.contextSize);
    const start = position - this.config.contextSize;

    for (let index = 0; index < this.config.contextSize; index++) {
      context[index] = this.indices[start + index];
    }

    return context;
  }

  buildInput(context) {
    const input = new Float32Array(this.inputSize);

    for (let slot = 0; slot < this.config.contextSize; slot++) {
      const charIndex = context[slot];
      const inputOffset = slot * this.config.embeddingSize;
      const embeddingOffset = charIndex * this.config.embeddingSize;

      for (let index = 0; index < this.config.embeddingSize; index++) {
        input[inputOffset + index] = this.embeddings[embeddingOffset + index];
      }
    }

    return input;
  }

  forward(context, temperature = 1) {
    const input = this.buildInput(context);
    const hidden = new Float32Array(this.config.hiddenSize);
    const logits = new Float32Array(this.vocabSize);

    for (let hiddenIndex = 0; hiddenIndex < this.config.hiddenSize; hiddenIndex++) {
      let sum = this.hiddenBias[hiddenIndex];

      for (let inputIndex = 0; inputIndex < this.inputSize; inputIndex++) {
        sum +=
          input[inputIndex] *
          this.hiddenWeights[inputIndex * this.config.hiddenSize + hiddenIndex];
      }

      hidden[hiddenIndex] = Math.tanh(sum);
    }

    for (let vocabIndex = 0; vocabIndex < this.vocabSize; vocabIndex++) {
      let sum = this.outputBias[vocabIndex];

      for (let hiddenIndex = 0; hiddenIndex < this.config.hiddenSize; hiddenIndex++) {
        sum +=
          hidden[hiddenIndex] *
          this.outputWeights[hiddenIndex * this.vocabSize + vocabIndex];
      }

      logits[vocabIndex] = sum / temperature;
    }

    return {
      input,
      hidden,
      probabilities: this.softmax(logits),
    };
  }

  softmax(logits) {
    const probabilities = new Float32Array(logits.length);
    let max = -Infinity;
    let total = 0;

    for (let index = 0; index < logits.length; index++) {
      max = Math.max(max, logits[index]);
    }

    for (let index = 0; index < logits.length; index++) {
      const value = Math.exp(logits[index] - max);
      probabilities[index] = value;
      total += value;
    }

    for (let index = 0; index < probabilities.length; index++) {
      probabilities[index] /= total;
    }

    return probabilities;
  }

  trainOne(position, learningRate) {
    const context = this.readContext(position);
    const target = this.indices[position];
    const { input, hidden, probabilities } = this.forward(context);
    const loss = -Math.log(Math.max(probabilities[target], 1e-9));
    const outputGradient = new Float32Array(probabilities);
    outputGradient[target] -= 1;

    const hiddenGradient = new Float32Array(this.config.hiddenSize);
    for (let hiddenIndex = 0; hiddenIndex < this.config.hiddenSize; hiddenIndex++) {
      let sum = 0;

      for (let vocabIndex = 0; vocabIndex < this.vocabSize; vocabIndex++) {
        sum +=
          this.outputWeights[hiddenIndex * this.vocabSize + vocabIndex] *
          outputGradient[vocabIndex];
      }

      hiddenGradient[hiddenIndex] = sum * (1 - hidden[hiddenIndex] ** 2);
    }

    const inputGradient = new Float32Array(this.inputSize);
    for (let inputIndex = 0; inputIndex < this.inputSize; inputIndex++) {
      let sum = 0;

      for (let hiddenIndex = 0; hiddenIndex < this.config.hiddenSize; hiddenIndex++) {
        sum +=
          this.hiddenWeights[inputIndex * this.config.hiddenSize + hiddenIndex] *
          hiddenGradient[hiddenIndex];
      }

      inputGradient[inputIndex] = sum;
    }

    for (let hiddenIndex = 0; hiddenIndex < this.config.hiddenSize; hiddenIndex++) {
      for (let vocabIndex = 0; vocabIndex < this.vocabSize; vocabIndex++) {
        const weightIndex = hiddenIndex * this.vocabSize + vocabIndex;
        this.outputWeights[weightIndex] -=
          learningRate * hidden[hiddenIndex] * outputGradient[vocabIndex];
      }
    }

    for (let vocabIndex = 0; vocabIndex < this.vocabSize; vocabIndex++) {
      this.outputBias[vocabIndex] -= learningRate * outputGradient[vocabIndex];
    }

    for (let inputIndex = 0; inputIndex < this.inputSize; inputIndex++) {
      for (let hiddenIndex = 0; hiddenIndex < this.config.hiddenSize; hiddenIndex++) {
        const weightIndex = inputIndex * this.config.hiddenSize + hiddenIndex;
        this.hiddenWeights[weightIndex] -=
          learningRate * input[inputIndex] * hiddenGradient[hiddenIndex];
      }
    }

    for (let hiddenIndex = 0; hiddenIndex < this.config.hiddenSize; hiddenIndex++) {
      this.hiddenBias[hiddenIndex] -= learningRate * hiddenGradient[hiddenIndex];
    }

    for (let slot = 0; slot < this.config.contextSize; slot++) {
      const charIndex = context[slot];
      const inputOffset = slot * this.config.embeddingSize;
      const embeddingOffset = charIndex * this.config.embeddingSize;

      for (let index = 0; index < this.config.embeddingSize; index++) {
        this.embeddings[embeddingOffset + index] -=
          learningRate * inputGradient[inputOffset + index];
      }
    }

    return loss;
  }

  generate(seed, length, temperature) {
    const output = seed || this.text.slice(0, this.config.contextSize);
    const generated = [...output];
    const context = new Uint16Array(this.config.contextSize);

    for (let index = 0; index < this.config.contextSize; index++) {
      context[index] = 0;
    }

    for (const char of [...output].slice(-this.config.contextSize)) {
      this.shiftContext(context, this.charToIndex.get(char) ?? 0);
    }

    for (let index = 0; index < length; index++) {
      const { probabilities } = this.forward(context, temperature);
      const nextIndex = this.sample(probabilities);
      const nextChar = this.vocab[nextIndex];
      generated.push(nextChar);
      this.shiftContext(context, nextIndex);
    }

    return generated.join('');
  }

  shiftContext(context, nextIndex) {
    for (let index = 0; index < context.length - 1; index++) {
      context[index] = context[index + 1];
    }

    context[context.length - 1] = nextIndex;
  }

  sample(probabilities) {
    const random = Math.random();
    let cumulative = 0;

    for (let index = 0; index < probabilities.length; index++) {
      cumulative += probabilities[index];

      if (random <= cumulative) {
        return index;
      }
    }

    return probabilities.length - 1;
  }
}

function getNumber(element, fallback) {
  const value = Number(element.value);
  return Number.isFinite(value) ? value : fallback;
}

function getConfig() {
  return {
    epochs: Math.max(1, Math.floor(getNumber(elements.epochsInput, 18))),
    contextSize: Math.max(2, Math.floor(getNumber(elements.contextInput, 14))),
    embeddingSize: Math.max(2, Math.floor(getNumber(elements.embeddingInput, 8))),
    hiddenSize: Math.max(8, Math.floor(getNumber(elements.hiddenInput, 48))),
    learningRate: Math.max(0.0001, getNumber(elements.learningRateInput, 0.045)),
    samplesPerEpoch: Math.max(50, Math.floor(getNumber(elements.samplesInput, 1600))),
  };
}

function normalizeText(text) {
  return text.replace(/\r\n/g, '\n').trim();
}

function updateTextStats() {
  const text = normalizeText(elements.trainingText.value);
  const uniqueCount = new Set(text).size;
  elements.textStats.textContent = `${text.length.toLocaleString()} characters · ${uniqueCount.toLocaleString()} unique symbols`;
}

function setTrainingState(isTraining) {
  elements.trainButton.disabled = isTraining;
  elements.stopButton.disabled = !isTraining;
  elements.generateButton.disabled = isTraining || !state.model;
}

function setProgress(percent, text) {
  elements.progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  elements.progressText.textContent = text;
}

function addLossLog(epoch, loss) {
  const row = document.createElement('div');
  row.className = 'loss-row';
  row.innerHTML = `<span>Epoch ${epoch}</span><strong>loss ${loss.toFixed(4)}</strong>`;
  elements.lossLog.prepend(row);
}

function nextFrame() {
  return new Promise(resolve => {
    requestAnimationFrame(resolve);
  });
}

function validateTrainingText(text, config) {
  if (text.length < config.contextSize + 2) {
    throw new Error('Add more text than the context size.');
  }

  if (new Set(text).size < 2) {
    throw new Error('The text needs at least two unique symbols.');
  }
}

async function trainModel() {
  const config = getConfig();
  const text = normalizeText(elements.trainingText.value);

  validateTrainingText(text, config);
  state.stopRequested = false;
  state.model = new ScratchTextModel(text, config);
  elements.lossLog.innerHTML = '';
  elements.outputText.textContent = 'Training...';
  elements.modelStatus.textContent = `Fresh random model · ${state.model.vocabSize} symbols`;
  setTrainingState(true);

  const positions = state.model.getSamplePositions(config.samplesPerEpoch);
  const totalSteps = positions.length * config.epochs;
  let completedSteps = 0;

  for (let epoch = 1; epoch <= config.epochs; epoch++) {
    if (state.stopRequested) {
      break;
    }

    const epochPositions = [...positions];
    state.model.shuffle(epochPositions);
    let lossTotal = 0;
    let samplesProcessed = 0;

    for (let sampleIndex = 0; sampleIndex < epochPositions.length; sampleIndex++) {
      if (state.stopRequested) {
        break;
      }

      lossTotal += state.model.trainOne(
        epochPositions[sampleIndex],
        config.learningRate
      );
      samplesProcessed++;
      completedSteps++;

      if (sampleIndex % 40 === 0) {
        const percent = (completedSteps / totalSteps) * 100;
        setProgress(
          percent,
          `Epoch ${epoch}/${config.epochs} · sample ${sampleIndex + 1}/${epochPositions.length}`
        );
        await nextFrame();
      }
    }

    if (samplesProcessed > 0) {
      const averageLoss = lossTotal / samplesProcessed;
      addLossLog(epoch, averageLoss);
      setProgress(
        (completedSteps / totalSteps) * 100,
        `Epoch ${epoch}/${config.epochs} complete · loss ${averageLoss.toFixed(4)}`
      );
    }
    await nextFrame();
  }

  const stopped = state.stopRequested;
  state.stopRequested = false;
  setTrainingState(false);
  elements.generateButton.disabled = !state.model;

  if (stopped) {
    setProgress((completedSteps / totalSteps) * 100, 'Training stopped.');
    elements.outputText.textContent =
      'Training was stopped. You can still generate from the partially trained model.';
  } else {
    setProgress(100, 'Training complete.');
    elements.modelStatus.textContent = `Trained · ${state.model.vocabSize} symbols · ${config.epochs} epochs`;
    generateText();
  }
}

function generateText() {
  if (!state.model) {
    elements.outputText.textContent = 'Train a model first.';
    return;
  }

  const length = Math.max(1, Math.floor(getNumber(elements.lengthInput, 500)));
  const temperature = Math.max(0.05, getNumber(elements.temperatureInput, 0.85));
  const seed = elements.seedInput.value;

  elements.outputText.textContent = state.model.generate(seed, length, temperature);
}

async function loadFile(file) {
  const text = await file.text();
  elements.trainingText.value = text;
  updateTextStats();
}

elements.trainingText.value = exampleText;
updateTextStats();

elements.fileInput.addEventListener('change', event => {
  const [file] = event.target.files;

  if (file) {
    loadFile(file);
  }
});

elements.trainingText.addEventListener('input', updateTextStats);
elements.trainButton.addEventListener('click', () => {
  trainModel().catch(error => {
    setTrainingState(false);
    setProgress(0, error.message);
    elements.outputText.textContent = error.message;
  });
});
elements.stopButton.addEventListener('click', () => {
  state.stopRequested = true;
});
elements.generateButton.addEventListener('click', generateText);
