# Scratch Text AI
This is a browser-based text generation lab that trains a small character-level neural network from scratch. It does not use OpenAI, pretrained weights, or any external model API: the model starts from random numbers and learns only from the text you paste or upload.

## Features

- User-friendly interface
- Paste text or upload a `.txt` file as a training corpus
- Configure epochs, context size, embedding size, hidden neurons, learning rate, and samples per epoch
- Train a tiny neural network locally in the browser with live loss/progress updates
- Generate new text from a prompt after training
- No server-side inference and no pretrained model weights

## Technologies Used
The application was built using the following technologies:

- HTML
- CSS
- JavaScript
- Vite

## Installation
To run this application on your local machine, you'll need to have Node.js installed. Once you have Node.js installed, follow these steps:

1. Clone the repository
2. Navigate to `client`
3. Run `npm install` to install the dependencies
4. Run `npm run dev` to start the development server
5. Open your browser and navigate to the Vite URL printed in the terminal

## Usage
Paste text into the training area or load a `.txt` file. Choose training settings, click **Train from scratch**, then use the generation panel to enter a prompt, choose length/temperature, and generate text.

This is an educational MVP rather than a production LLM. Larger text files, more epochs, and consistent writing style generally produce better results, but training still happens entirely in your browser.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Push your changes to your forked repository
6. Submit a pull request
Please make sure to include a detailed description of your changes in your pull request.

## License
This project is licensed under the MIT license. See the LICENSE file for more information.

## Contact
If you have any questions or comments about this project, please feel free to contact me at [andrewtsegaye@jsmastery.pro](mailto:andrewtsegaye7@gmail.com).
