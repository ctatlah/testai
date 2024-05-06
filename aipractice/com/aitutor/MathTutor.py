'''
Created on May 6, 2024

Chat GPT math tutor client

@author: ctatlah
'''

import time
import os
import openai
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler

def get_openai_key():
    """
    Hack to get the key from .env file.
    TODO : Look into why environ not getting key set in .zprofile and
           why OpenAI provided solution didnt work.
    Args:
      none
    Returns:
      String : openai api key
    """
    #key = os.environ.get("OPENAI_API_KEY")
    f = open(".env", "r")
    key = f.read().split("=")[1]   
    return key
    
def chatgpt_response(client, assistant, thread, uinput):
    """
    Chat interaction with the assistant.
    Args:
      client    : openai client
      assistant : chatgpt math tutor assistant
      thread    : client thread, conversation between user and and assistant
      uinput    : users query
    Returns:
      status (string) : status of run: 'completed' or 'failed'
    """
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=uinput
        )
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please help answer elementary students questions"
    )
    
    print("Thinking...", end="")
    while True:
        time.sleep(1)
        current_run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if current_run.status == "completed":
            print("Got an answer!")
            response_message = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            print(response_message)
            break
        elif current_run.status == "failed":
            print("Something went wrong!")
            print(current_run.last_error.message)
            break
        else:        
            print(".", end="")
            
    return current_run.status
    

def main():
    """
    Chat interaction with math assistant.
    """
    
    # Setup ChatGPT client
    #
    client = OpenAI(api_key=get_openai_key())
    
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor for elementary aged children. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-3.5-turbo",
        )
    
    thread = client.beta.threads.create()
    
    # Lets answer some questions
    #
    print("\n")
    print("Hi I'm your personal math tutor.", end=" ")
    print("What math problem can I help you with today?", end=" ")
    print("(press q to quit)\n")
    
    while True:
        user_input = input("What would you like to ask me? ")
        if user_input.lower() == "q":
            print("I hope I was able to help you out today! Goodbye.")
            break
        
        status = chatgpt_response(client, assistant, thread, user_input)
        
        if status == "failed":
            break

if __name__ == "__main__":
    main()