import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

app.get('/', async (req, res) => {
  res.status(200).send({
    message:
      'Scratch Text AI trains locally in the browser and does not use external model APIs.',
  });
});

app.post('/', async (req, res) => {
  res.status(410).send({
    message:
      'Text generation moved to the browser. Train a new local model from the UI.',
  });
});

app.listen(5000, () =>
  console.log('Scratch Text AI helper server started on http://localhost:5000')
);
