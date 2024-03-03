# RAG-App-with-Holoviz-Panel
Different ways to build a multiple document querying app with Holoviz Panel

## 1. App with RetrievalQA and Chain of Thought
This uses the `langchain.chains.RetrievalQA` method to query on the vector DB, and `panel.chat.langchain.PanelCallbackHandler` to display the chain of thought process behind generating LLM responses. That is, it displays:
- It shows the source documents
- It streams the response from the LLM
- And when the streaming from step 2 ends, it once again displays the final answer below it.
- Changes the name of the chatbot at each step above, depending on what function it is doing.

Unfortunately, there's no way (at least as of panel version '1.3.8') to configure the callback handler to only stream the response, and hide the source documents and not display the final answer after streaming.

## 2. App with openai.chat.completions
This uses langchain only to create the vector DB, and for the querying process uses the `openai.chat.completions` module. However, this does not stream the response in the panel UI, and instead displays only the final answer. Since we are not using langchain methods for querying on the vector DB with the LLM, we have to explicitly code the process of sending the context (i.e., the relevant docs/chunks retrieved from the vector DB) and append the chat history after every user query and LLM response.

## 3. App with openai.chat.completions and streaming response
This streams the LLM response. For this, we need to remove the `async response` function from the previous version, and instead modify the `get_LLM_response` function as `async` and yield the LLM response token by token, and send that as callback to the panel chat interface.
