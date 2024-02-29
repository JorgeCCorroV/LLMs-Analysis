I tried to implement the analysis on the Harry Potter and The Philosofer’s Stone book. It seemed a good practice to understand more about NLP and LLMs. I gained hands- on experience with several key concepts:
•	Preprocessing raw text into a format usable for machine learning models.
•	Implementing a basic bigram model in PyTorch by having the model output a softmax over next character logits
•	Generating text by recursively sampling from the model to obatain the next most likely character.

While conceptually straightforward, I could encounter some challenges in implementation:
•	I changed manually the hyperparameters of the model.
•	The generated samples lacked global coherency.

I would have liked to have the time to extend the analysis and apply something similar but different. For example:
•	Implement a RNN-based language model rather than simple bigram model to improve coherence.
•	Instead of trying train, test and predict by characters, doing it by words.
