/**
 * --- MATRIX ENGINE ---
 * Hand-written linear algebra for neural computations.
 */
class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
    }

    static fromArray(arr) {
        let m = new Matrix(arr.length, 1);
        for (let i = 0; i < arr.length; i++) m.data[i][0] = arr[i];
        return m;
    }

    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) arr.push(this.data[i][j]);
        }
        return arr;
    }

    randomize() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let limit = Math.sqrt(6 / (this.rows + this.cols));
                this.data[i][j] = Math.random() * (limit * 2) - limit;
            }
        }
    }

    add(n) {
        if (n instanceof Matrix) {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) this.data[i][j] += n.data[i][j];
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) this.data[i][j] += n;
            }
        }
    }

    static subtract(a, b) {
        let res = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) res.data[i][j] = a.data[i][j] - b.data[i][j];
        }
        return res;
    }

    static multiply(a, b) {
        if (a.cols !== b.rows) return null;
        let res = new Matrix(a.rows, b.cols);
        for (let i = 0; i < res.rows; i++) {
            for (let j = 0; j < res.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) sum += a.data[i][k] * b.data[k][j];
                res.data[i][j] = sum;
            }
        }
        return res;
    }

    multiply(n) {
        if (n instanceof Matrix) {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) this.data[i][j] *= n.data[i][j];
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) this.data[i][j] *= n;
            }
        }
    }

    static transpose(matrix) {
        let res = new Matrix(matrix.cols, matrix.rows);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) res.data[j][i] = matrix.data[i][j];
        }
        return res;
    }

    map(func) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) this.data[i][j] = func(this.data[i][j]);
        }
    }

    static map(matrix, func) {
        let res = new Matrix(matrix.rows, matrix.cols);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) res.data[i][j] = func(matrix.data[i][j]);
        }
        return res;
    }
}

/**
 * --- NEURAL NETWORK ---
 * MLP with Backpropagation algorithm.
 */
class NeuralNetwork {
    constructor(in_nodes, hid_nodes, out_nodes) {
        this.input_nodes = in_nodes;
        this.hidden_nodes = hid_nodes;
        this.output_nodes = out_nodes;

        this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
        this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
        this.weights_ih.randomize();
        this.weights_ho.randomize();

        this.bias_h = new Matrix(this.hidden_nodes, 1);
        this.bias_o = new Matrix(this.output_nodes, 1);
        this.bias_h.randomize();
        this.bias_o.randomize();

        this.learning_rate = 0.15;
    }

    sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
    dsigmoid(y) { return y * (1 - y); }

    predict(input_array) {
        let inputs = Matrix.fromArray(input_array);
        let hidden = Matrix.multiply(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        hidden.map(this.sigmoid);

        let output = Matrix.multiply(this.weights_ho, hidden);
        output.add(this.bias_o);
        output.map(this.sigmoid);
        return output.toArray();
    }

    train(input_array, target_array) {
        let inputs = Matrix.fromArray(input_array);
        let hidden = Matrix.multiply(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        hidden.map(this.sigmoid);

        let outputs = Matrix.multiply(this.weights_ho, hidden);
        outputs.add(this.bias_o);
        outputs.map(this.sigmoid);

        let targets = Matrix.fromArray(target_array);
        let output_errors = Matrix.subtract(targets, outputs);

        let gradients = Matrix.map(outputs, this.dsigmoid);
        gradients.multiply(output_errors);
        gradients.multiply(this.learning_rate);

        let hidden_T = Matrix.transpose(hidden);
        let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);
        this.weights_ho.add(weight_ho_deltas);
        this.bias_o.add(gradients);

        let who_T = Matrix.transpose(this.weights_ho);
        let hidden_errors = Matrix.multiply(who_T, output_errors);

        let hidden_gradient = Matrix.map(hidden, this.dsigmoid);
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.learning_rate);

        let inputs_T = Matrix.transpose(inputs);
        let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);
        this.weights_ih.add(weight_ih_deltas);
        this.bias_h.add(hidden_gradient);

        let sumErr = 0;
        output_errors.toArray().forEach(e => sumErr += e*e);
        return sumErr / this.output_nodes;
    }
}

/**
 * --- DATASET & UI LOGIC ---
 */
function generateDigitPattern(digit) {
    let p = new Array(784).fill(0);
    const center = 14;
    if (digit === 0) {
        for(let r=6; r<22; r++) {
            for(let c=6; c<22; c++) {
                let dist = Math.sqrt((r-center)**2 + (c-center)**2);
                if(dist > 5 && dist < 8) p[r*28+c] = 1;
            }
        }
    } else if (digit === 1) {
        for(let r=4; r<24; r++) { p[r*28+14] = 1; p[r*28+15] = 1; }
    } else if (digit === 2) {
        for(let c=8; c<20; c++) { p[6*28+c]=1; p[22*28+c]=1; }
        for(let i=0; i<16; i++) {
            let r = 6 + i; let c = 19 - i;
            p[r*28+c] = 1;
        }
    }
    return p;
}

const demoData = [
    { target: [1, 0, 0], pixels: generateDigitPattern(0) },
    { target: [0, 1, 0], pixels: generateDigitPattern(1) },
    { target: [0, 0, 1], pixels: generateDigitPattern(2) }
];

let nn = new NeuralNetwork(784, 48, 3);
let isTraining = false;
let history = [];

const canvas = document.getElementById('drawing-board');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const visCanvas = document.getElementById('vis-canvas');
const vctx = visCanvas.getContext('2d');
let drawing = false;

function init() {
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    
    const updateSize = () => {
        visCanvas.width = visCanvas.offsetWidth;
        visCanvas.height = visCanvas.offsetHeight;
        drawVisualization();
    };
    window.onresize = updateSize;
    updateSize();

    canvas.onmousedown = () => drawing = true;
    window.onmouseup = () => { drawing = false; ctx.beginPath(); };
    canvas.onmousemove = (e) => {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    };
    updateProbBars([0,0,0]);
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updateProbBars([0,0,0]);
}

function predictDrawing() {
    const temp = document.createElement('canvas');
    temp.width = 28; temp.height = 28;
    temp.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
    const data = temp.getContext('2d').getImageData(0,0,28,28).data;
    let inputs = [];
    for(let i=0; i<data.length; i+=4) inputs.push(data[i]/255);
    
    const out = nn.predict(inputs);
    updateProbBars(out);
    drawVisualization(inputs);
}

function updateProbBars(probs) {
    const container = document.getElementById('prob-bars');
    container.innerHTML = probs.map((p, i) => `
        <div class="bar-container">
            <span class="bar-label">${i}</span>
            <div class="bar-outer"><div class="bar-inner" style="width: ${p*100}%"></div></div>
            <span style="font-size:0.65rem; color:var(--text-dim); width:35px">${(p*100).toFixed(0)}%</span>
        </div>
    `).join('');
}

async function startTraining() {
    if (isTraining) return;
    isTraining = true;
    const btn = document.getElementById('train-btn');
    btn.disabled = true;
    document.getElementById('status').innerText = 'Computing...';

    for (let e = 0; e < 200; e++) {
        let err = 0;
        demoData.forEach(d => err += nn.train(d.pixels, d.target));
        history.push(err/3);
        if (history.length > 100) history.shift();

        if (e % 5 === 0) {
            document.getElementById('epoch-count').innerText = e + 1;
            document.getElementById('loss-val').innerText = (err/3).toFixed(5);
            drawLossChart();
            await new Promise(r => setTimeout(r, 0));
        }
    }
    document.getElementById('status').innerText = 'Optimization Complete';
    btn.disabled = false;
    isTraining = false;
}

function drawLossChart() {
    const c = document.getElementById('loss-chart');
    const ct = c.getContext('2d');
    ct.clearRect(0,0,c.width,c.height);
    ct.strokeStyle = '#00f2ff';
    ct.lineWidth = 2;
    ct.beginPath();
    history.forEach((v, i) => {
        let x = (i / history.length) * c.width;
        let y = c.height - (v * c.height * 5);
        if (i === 0) ct.moveTo(x, y); else ct.lineTo(x, y);
    });
    ct.stroke();
}

function drawVisualization(activeInputs = null) {
    vctx.clearRect(0, 0, visCanvas.width, visCanvas.height);
    const nodes = [12, 8, 3];
    const xPos = [40, visCanvas.width/2, visCanvas.width-40];
    
    for (let i=0; i<nodes[0]; i++) {
        let y1 = (visCanvas.height/(nodes[0]+1))*(i+1);
        for (let j=0; j<nodes[1]; j++) {
            let y2 = (visCanvas.height/(nodes[1]+1))*(j+1);
            let w = nn.weights_ih.data[j][i];
            let alpha = Math.abs(w) * 0.5;
            vctx.strokeStyle = w > 0 ? `rgba(0, 242, 255, ${alpha})` : `rgba(218, 54, 51, ${alpha})`;
            vctx.beginPath(); vctx.moveTo(xPos[0], y1); vctx.lineTo(xPos[1], y2); vctx.stroke();
        }
    }

    nodes.forEach((count, l) => {
        for (let i=0; i<count; i++) {
            let y = (visCanvas.height/(count+1))*(i+1);
            vctx.fillStyle = "#161b22";
            vctx.beginPath(); vctx.arc(xPos[l], y, 6, 0, Math.PI*2); vctx.fill();
            vctx.strokeStyle = "#30363d";
            vctx.stroke();
            if (l === 2) {
                vctx.fillStyle = "#8b949e";
                vctx.fillText(i, xPos[l]+12, y+4);
            }
        }
    });
}

window.onload = init;
