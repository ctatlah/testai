'''
Created on May 6, 2024

Web based Chat GPT math tutor client

@author: ctatlah
'''

import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, template_folder='../pages/templates')

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route("/MathTutorWeb", methods=["POST"])
def MathTutorWeb():
    # Get the user message from the request form
    user_message = request.form["message"]

    # Get response from ChatGPT assistant and send it back
    response_msg, response_err = send_to_chatgpt(user_message)
    
    if response_msg != None:
        return jsonify({"message": response_msg})
    else:
        return jsonify({"error": response_err}), 500

def send_to_chatgpt(user_query):
    """
    Chat interaction with math assistant.
    Args:
      user_query (string) : users question to ask ChatGPT assistant
    Returns:
      response_message : response from ChatGPT
      response_error   : error message from ChatGPT
    """
    client, assistant, thread = create_chat_gpt_client()
    return chatgpt_response(client, assistant, thread, user_query)
    
def create_chat_gpt_client():
    """
    ChatGPT client setup. Creates client, assistant and thread
    Args:
      none
    Returns:
      client    : OpenAI client
      assistant : ChatGPT assistant
      thread    : Message thread
    """
    # Get OpenAI api keys
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    
    # Setup ChatGPT client
    #
    client = OpenAI(api_key=key)
    
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor for elementary aged children. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-3.5-turbo",
        )
    
    thread = client.beta.threads.create()
    
    return client, assistant, thread

def chatgpt_response(client, assistant, thread, user_query):
    """
    Send query to OpenAI and return response
    Args:
      client    : openai client
      assistant : chatgpt math tutor assistant
      thread    : client thread, conversation between user and and assistant
      uinput    : users query
    Returns:
      response_message (string) : response from ChatGPT client
      response_error (String)   : any response errors from ChatGPT client
    """
    response_message = None
    response_error = None
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_query
        )
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please help answer elementary students questions"
    )
    
    while True:
        time.sleep(1)
        current_run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if current_run.status == "completed":
            response_message = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            break
        elif current_run.status == "failed":
            response_error = current_run.last_error.message
            break
        else:        
            continue
            
    return response_message, response_error

if __name__ == "__main__":
    app.run(debug=True)