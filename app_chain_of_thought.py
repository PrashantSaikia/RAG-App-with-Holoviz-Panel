import os, dotenv, openai, panel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

panel.extension()

# Set API key
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

@panel.cache
def load_vectorstore():
    # If the vector embeddings of the documents have not been created
    if not os.path.isfile('chroma_db/chroma.sqlite3'):

        # Load the documents
        loader = DirectoryLoader('Docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
        data = loader.load()

        # Split the docs into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )
        docs = splitter.split_documents(data)

        # Embed the documents and store them in a Chroma DB
        embedding=OpenAIEmbeddings(openai_api_key = openai.api_key)
        vectorstore = Chroma.from_documents(documents=docs,embedding=embedding, persist_directory="./chroma_db")
    else:
        # load ChromaDB from disk
        embedding=OpenAIEmbeddings(openai_api_key = openai.api_key)
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

    return vectorstore


def retrieval_qa_chain():

    # Define prompt template
    template = """
    **Objectives**: Generate a smart contract in Compact language for a given task, adhering to the grammar and abstract data types (ADTs) of Compact. In addition to generating code, you may be asked to describe concepts, functionalities, or provide explanations on various aspects of the Compact programming language, smart contract logic, or blockchain principles. If you are asked to generate compact code, here are the guidelines:

    ## Requirements

    1. **Starting Point**: All Compact code must begin with `include "std";`.
    2. **Counter ADT Usage**:
       - Use `Counter` for increment and decrement operations.
       - Direct retrieval of `Counter` value or setting it to a specific value is not supported.
       - Initialization of a `Counter` variable does not require a constructor. Define it directly in the ledger block as `count: Counter;`.
    3. **Type Naming**:
       - Use full type names, e.g., `Unsigned Integer[16]`, not `Uint16`.
    4. **Function Definitions**:
       - Use `export circuit` for functions that change the ledger's state.
       - Mark other public functions or types with `export`.
    5. **Error Handling**: Use `assert` statements for error handling.
    6. **Ledger Access**: Access ledger fields with the `ledger` keyword.
    7. **Special Instructions**:
       - Initialize `Map` and `Set` types using the `insert` method or as empty literals.
       - Do not use semicolons in for loops.
       - The `>` and `<` symbols are not defined. Use `Counter` for related functionalities.
       - `Counter` cannot be directly instantiated with a value. Use `increment` for initialization.
       - Use `reset_to_default()` for variable initialization inside constructor.
       - The `read()` method is the correct way to access a `Counter`'s value.
       - Implement a witness function like `local_voter_id()` to obtain off-chain data. Witness functions are designed to obtain data from the local machine without generating constraints in a circuit, which is suitable for retrieving a voter's unique ID from an off-chain source. This ensures that the voter's ID is kept confidential and is not exposed on the blockchain.
       - When declaring variables, let is used to declare a variable that can be assigned a value once and is mutable, meaning its value can be changed later in the code. 
       On the other hand, const is used to declare a constant whose value cannot be changed once it has been assigned. So, use them appropriately wherever needed.
       - The assert statement should not include a comma between the condition and the error message. 
       - Prefer using "if" instead of "match" for conditional statements whenever possible.
       - Whenever applicable, make one function do only one thing. For example, in a binary voting contract, instead of using one "get_vote_tally" function, implement two 
       functions, "get_vote_tally_yes" and "get_vote_tally_no" to get the voting count of the options, respectively.
       - The constructor, if needed, should always be defined within the ledger block. It should never be defined outisde the ledger block.
       - Ensure simplicity and adherence to Compact's syntax and semantics. For example, the keyword "function" is not defined. Instead, Compact uses the keyword "circuit" 
       to define what would traditionally be called functions in other programming languages.
       - Use "const" instead of "let" when declaring variables.
       - Syntax Guidelines for assert and if Statements:
        When generating Compact code or providing code examples, adhere to the correct syntax for assert and if statements

        assert Statements:
        Do not use parentheses or commas within assert statements. Write assert directly followed by the condition and the error message string without enclosing the condition in parentheses.
        Incorrect: assert(condition, "Error message");
        Correct: assert condition "Error message";
        if Statements:
        Always use parentheses around the condition in if statements. This helps in clearly defining the scope of the condition being evaluated.
        Incorrect: if condition
        Correct: if (condition)

    ## Task Context and History

    - **Context**: {context}
    - **Chat History**: {history}
    - **User Question**: {question}

    ## Answer Template

    Provide a Compact code snippet that meets the above requirements, keeping explanations concise and to the point.

    """

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    )

    llm = ChatOpenAI(temperature=0, 
                     model="gpt-4-1106-preview",
                     openai_api_key = openai.api_key,
                     streaming=True
                    )

    vectorstore = load_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(llm,
                                          chain_type='stuff',
                                          retriever=vectorstore.as_retriever(),
                                          chain_type_kwargs={
                                              "prompt": prompt,
                                              "memory": memory
                                          })
    return qa_chain


async def respond(contents, user, chat_interface):
    qa = retrieval_qa_chain()
    callback_handler = panel.chat.langchain.PanelCallbackHandler(chat_interface)
    yield qa(contents, callbacks=[callback_handler])['result']

chat_interface = panel.chat.ChatInterface(
    callback=respond, sizing_mode="stretch_width", callback_exception='verbose'
)
chat_interface.send(
    {"user": "CompactBot", "value": '''Welcome to CompactBot, your personal assistant for generating, troubleshooting, and understanding Compact code. I have been trained on Midnight's documentionation of the Compact smart contract programming language. Some guidenlines about what I can and cannot do have been provided in the Readme. The more detailed your question is, the better I can answer it.\n\nFor example, instead of asking me "𝙒𝙧𝙞𝙩𝙚 𝙢𝙚 𝙖 𝘾𝙤𝙢𝙥𝙖𝙘𝙩 𝙘𝙤𝙙𝙚 𝙩𝙤 𝙞𝙢𝙥𝙡𝙚𝙢𝙚𝙣𝙩 𝙖 𝙨𝙞𝙢𝙥𝙡𝙚 𝙫𝙤𝙩𝙞𝙣𝙜 𝙘𝙤𝙣𝙩𝙧𝙖𝙘𝙩", if you provide me with a bit more details on what functionalities exactly you want, that would help me greatly: "𝙒𝙧𝙞𝙩𝙚 𝘾𝙤𝙢𝙥𝙖𝙘𝙩 𝙘𝙤𝙙𝙚 𝙩𝙤 𝙞𝙢𝙥𝙡𝙚𝙢𝙚𝙣𝙩 𝙖 𝙨𝙞𝙢𝙥𝙡𝙚 𝙫𝙤𝙩𝙞𝙣𝙜 𝙘𝙤𝙣𝙩𝙧𝙖𝙘𝙩. 𝙏𝙝𝙚 𝙘𝙤𝙣𝙩𝙧𝙖𝙘𝙩 𝙨𝙝𝙤𝙪𝙡𝙙 𝙚𝙣𝙖𝙗𝙡𝙚 𝙖 𝙥𝙧𝙚𝙙𝙚𝙛𝙞𝙣𝙚𝙙 𝙡𝙞𝙨𝙩 𝙤𝙛 𝙥𝙖𝙧𝙩𝙞𝙘𝙞𝙥𝙖𝙣𝙩𝙨 𝙩𝙤 𝙫𝙤𝙩𝙚 𝙤𝙣 𝙖 𝙗𝙞𝙣𝙖𝙧𝙮 𝙘𝙝𝙤𝙞𝙘𝙚 (𝙚.𝙜., 𝙔𝙚𝙨/𝙉𝙤). 𝙀𝙖𝙘𝙝 𝙥𝙖𝙧𝙩𝙞𝙘𝙞𝙥𝙖𝙣𝙩 𝙞𝙨 𝙞𝙙𝙚𝙣𝙩𝙞𝙛𝙞𝙚𝙙 𝙗𝙮 𝙖 𝙪𝙣𝙞𝙦𝙪𝙚 𝙄𝘿 𝙖𝙣𝙙 𝙘𝙖𝙣 𝙫𝙤𝙩𝙚 𝙤𝙣𝙡𝙮 𝙤𝙣𝙘𝙚. 𝙏𝙝𝙚 𝙘𝙤𝙣𝙩𝙧𝙖𝙘𝙩 𝙨𝙝𝙤𝙪𝙡𝙙 𝙩𝙖𝙡𝙡𝙮 𝙫𝙤𝙩𝙚𝙨 𝙛𝙤𝙧 𝙗𝙤𝙩𝙝 𝙘𝙝𝙤𝙞𝙘𝙚𝙨 𝙖𝙣𝙙 𝙚𝙣𝙨𝙪𝙧𝙚 𝙩𝙝𝙚 𝙞𝙣𝙩𝙚𝙜𝙧𝙞𝙩𝙮 𝙖𝙣𝙙 𝙘𝙤𝙣𝙛𝙞𝙙𝙚𝙣𝙩𝙞𝙖𝙡𝙞𝙩𝙮 𝙤𝙛 𝙩𝙝𝙚 𝙫𝙤𝙩𝙞𝙣𝙜 𝙥𝙧𝙤𝙘𝙚𝙨𝙨."\n\nPlease ask me any question you might have about or related to Compact.'''},
    respond=False,
)

template = panel.template.BootstrapTemplate(main=[chat_interface])

template.servable()




