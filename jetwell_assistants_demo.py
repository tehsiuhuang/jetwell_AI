from openai import OpenAI
from time import sleep

# Set your OpenAI API key here
client = OpenAI(api_key='ask-me')


# Your assistant's ID
assistant_id = 'asst_LgTVbtxWqGSWe043emxNxKpN'

lv_prompt1 = ("什麼是大綜學苑?")

my_assistant = client.beta.assistants.retrieve(assistant_id)

thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": lv_prompt1
        }
    ]
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=my_assistant.id
)

while run.status != 'completed':
    run = client.beta.threads.runs.retrieve(
      thread_id=thread.id,
      run_id=run.id
    )
    print(run.status)
    sleep(5)


thread_messages = client.beta.threads.messages.list(thread.id)
print(thread_messages)
