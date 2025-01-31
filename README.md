# Conversational AI Chatbot Prototype

The Conversational AI Chatbot Prototype is a voice-based AI chatbot designed to enable natural and seamless interactions between users and the system. It leverages natural language understanding (NLU) and speech recognition to provide a robust and intelligent conversational experience.

## Project Architecture

![Architecture Diagram](Project_Diagrams/Activity%20Diagram.png)

## Features

**Voice-Based Interaction**: Users can communicate with the chatbot using voice input.

**Natural Language Processing (NLP)**: Utilizes NLP techniques to understand and generate responses.

**Speech-to-Text (STT) and Text-to-Speech (TTS)**: Converts user speech into text and vice versa for a fluid experience.

**Context Awareness**: Maintains conversation context for meaningful interactions.

## Project Output

<img src="Project_Output/output_1.png" width="800" height="280">
<img src="Project_Output/output_2.png" width="800" height="280">

## Required Libraries

TensorFlow / TFLearn (for deep learning-based NLP models)

Scikit-learn (for feature extraction and similarity calculations)

SpeechRecognition (for voice input processing)

NLTK / spaCy (for text processing)

gTTS / pyttsx3 (for text-to-speech conversion)

## Machine Learning Techniques Used

Deep Neural Networks (DNN): Implemented using TFLearn with multiple fully connected layers.

Bag-of-Words Model: Used for text representation and intent classification.

Cosine Similarity: Applied for text similarity calculations in response generation.

Tokenization & Stemming: Applied using NLTK and spaCy for text preprocessing.

Count Vectorization: Used for feature extraction from text with Scikit-learn.

## Installation

### Clone the repository:

```bash
git clone https://github.com/Patel4007/Conversational-AI-Prototype.git
cd Conversational-AI-Prototype
```

### Create and activate a virtual environment:

```bash
python -m venv venv
```

```bash
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the program:

```bash
python app.py
```

## Usage

Start the chatbot by running the program.

Use a microphone to interact with the chatbot via voice.

The chatbot processes speech input, generates a response and provides a spoken output.

## Future Enhancements

Support for multiple languages

Improved dialogue management with reinforcement learning

## Contribution

Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
