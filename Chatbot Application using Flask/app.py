from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Initialize Flask app with template folder
app = Flask(__name__, template_folder=os.getcwd())  # Use current working directory

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Route to serve the CSS file
@app.route('/style.css')
def serve_css():
    return send_from_directory(os.getcwd(), 'style.css')  # Serve the style.css file

# Route to render the chatbot page
@app.route("/")
def index():
    return render_template('chatbot.html')  # Flask will now find chatbot.html in the same directory as app.py

# Route for processing chat messages
@app.route("/get", methods=["GET", "POST"])
def chat():
    # Retrieve the message from the request (use request.form.get for POST requests)
    msg = request.form.get("msg") if request.method == "POST" else request.args.get("msg")
    
    if msg:
        input_text = msg
        return get_Chat_response(input_text)
    else:
        return jsonify({"error": "No message received"}), 400

# Function to generate chat response
def get_Chat_response(text):
    # Initialize chat_history_ids for the first call
    chat_history_ids = None

    # Let's chat for 5 lines
    for step in range(5):
        # Encode the new user input, add the eos_token, and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history (if chat history exists)
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generate a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Return the bot's last response
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
