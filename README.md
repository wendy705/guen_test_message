# ✨ Guendalina x TextCortex

This small repository includes my attempt to solve the TextCortex NLP Engineer Challenge.
A small refresher on the challenge, rules and deliverables before we start the actual documentation:

📐 Rules

* You cannot use an external API like OpenAIs GPT-3
* No limits on what kind of AI Model can be used but it should fit to a single nVidia T4 GPU.
* The generative model should be able to run on a T4 GPU with 16GB of Memory
* If you want to train a model, you can use Google Colab for free if this is not enough
for your training needs, we can also give access to our cloud resources.
* The model should be accessible by a REST API Call.
* If you decide to use our compute resoureces, it should only be used for developing
  the MVP. Any other use will automatically disqualify you from this challenge.

🚚 Deliverables (with my answers as to where to find them) ✅
* A github repo with the source code to run, data and model can be uploaded to
huggingface or somewhere else. 
  * ✅ If you are reading this documentation, you should have access to the repository :)
  * ✅ The model has been uploaded to the HuggingFace model hub in private view, and the dataset was already part of the HuggingFace datasets library
* An endpoint that is able to write a message by consuming inputs and using them in
the actual message (think about from, to, key action points...).
  * ✅ You will find a section in this document detailing how to access the endpoint :)
* A brief documentation describing your thought and learning processes.
  * ✅ You are reading it ;)

## 🤔 My Thought and Learning Process 

### 1. Selecting a Dataset

I don't remember where, I read something that really stuck with me. "In Data Science Data comes before the Science". Data is the key to a successful model. Unfortunately, data is at the same time abundant and scarce. While we have an enormous amount of data available, sometimes we can't find exactly what we want. So we need to get a bit creative.

The aim of this exercise, if I understood correctly, is to give the machine a prompt. e.g. the key points of the message we want to send/the text we want to write, and have the model write the rest for us. Thinking about it, it is almost as if we wanted to write the subject of our email, and then have the email write itself - if only! 

So I decided to use the aeslc dataset. The dataset comprises email and subject line pairs. Normally, the email is given as the input, and we want the model to extract the subject line. In this case, we are switching things up. 

One of the advantages of this dataset is that it is already available in the HuggingFace datasets library. So it is already clean, nicely formatted, and ready to use.

###2. Select a model from the HuggingFace Transformers library and Load the Data

Now that we have our input and output data, we are going to select a model from the HuggingFace transformer library. This allows us to use a pre-trained model, that has been trained on a large amount of data to perform several different tasks. This is what we call the body of a model. We are then going to fine-tune it for a specific task, meaning we are going to create the head of our model.

For this test, we are going to use the [Google T5 model](https://huggingface.co/t5-base). You can use its multilingual version, MT5, to train your model in another language. The MT5 model supports 101 languages. The T5-base model has 220 million parameters, which means it is quite light compared to other models, but at the same time it is flexible and powerful enough to learn new tasks with simple fine-tuning ([Google AI blog](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)).

### 3. Fine-Tuning the Model to create a new Head

Here we define the model and start setting everything up for our fine-tuning process.

As you can see in the code, there are a few steps we need to take in order to make sure we have all the pieces necessary to train our transformer.
Comments in the code will help you navigate the different functions if necessary, but to put it briefly:
* we are going to import libraries and tools we need to carry out the process
* we are going to set a seed to make sure our experiments can be easily reproduced
* we are going to define a fineTuner class, that extends the lightning module by PyTorch, in order to carry out the fine-tuning
* we are going to set the arguments necessary to train our model, such us batch size, number of epochs, number of gpus used, etc.
* we are going to create a class that extends Pytorch's Dataset, in order to correctly process our data and build our input and target tensors

Once we have done all of these steps, we can start training the model and experimenting.
Once we are happy with our results, we can push the model on the huggingFace model hub, and create our endpoint.

## 🙋 How to access the endpoint?

First, you need to start the server. You can do so by just running the following command:
```bash
$> python ./server.py
```

* You can access the endpoint directly from your terminal using a curl command such as:
```bash
$> curl --location --request POST 'http://127.0.0.1:8008/' \
--header 'Content-Type: application/json' \
--data-raw '{
    "prompt": "we will have a system update on friday. please backup your work",
    "token_count": 250,
    "temperature": 0.7,
    "n_gen": 6
}'
```
* You can access the endpoint through an interface such as Postman
* You can access the endpoint through a Python http request as follows:
```python
import http.client
import json

conn = http.client.HTTPSConnection("127.0.0.1", 8008)
payload = json.dumps({
  "prompt": "we will have a system update on friday. please backup your work",
  "token_count": 250,
  "temperature": 0.7,
  "n_gen": 6
})
headers = {
  'Content-Type': 'application/json'
}
conn.request("POST", "/", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
```
* You can run the rest-api.http script you will find in this repository 😊 

## ⭐ Et voilà!

You should have all the information needed, but please don't hesitate to contact me if you have any issues or questions regarding any of the expected deliverables.
Thank you for the opportunity, this was a really fun exercise, and I will definitely play with the model more!
😊