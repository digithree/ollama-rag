from tinydb import TinyDB
import os, re, shutil, subprocess

MAIN_MODEL_DEFAULT = "gemma:7b"
FAST_MODEL_DEFAULT = "mistral"
MAIN_MODEL_NAME = "ragmain"

MODELFILE_TEMPLATE = "Modelfile-template"
MODELFILE_GENERATED = "Modelfile-generated"

db = TinyDB('./config.json')
agent_table = db.table('agent')
model_table = db.table('model')

def start():
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # required to run Chroma DB properly on CPU
    print("*** CONFIG TOOL")
    configAgent()
    configModel()
    writeModelfile()
    createModel()
    print("*** CONFIG IS COMPLETE")

def configAgent():
    row = None
    if len(agent_table.all()) > 0:
        row = agent_table.all()[0]
    if row == None:
        row = {"active": True}
        agent_table.insert(row)
    if row.get("agent_type") == None:
        user_input = inputForAccepted(
            "Chatbot agent type/description:",
            lambda: input("E.g. \"a research assistant\"\n> The chatbot agent is... "),
            lambda _: print("Accept?")
        ).strip()
        row["agent_type"] = user_input
        agent_table.update(row, doc_ids=[1])
    if row.get("agent_name") == None:
        user_input = inputForAccepted(
            "Agent (chatbot) name (a single first name works best):",
            lambda: input("> Name: "),
            lambda _: print("Accept?")
        ).strip()
        row["agent_name"] = user_input
        agent_table.update(row, doc_ids=[1])
    if row.get("agent_relation") == None:
        user_input = inputForAccepted(
            "Agent relation (writen as to the agent):",
            lambda: input(f"E.g. \"your supervisor\"\n> I am... "),
            lambda _: print("Accept?")
        ).strip()
        row["agent_relation"] = user_input
        agent_table.update(row, doc_ids=[1])
    if row.get("agent_attitude") == None:
        user_input = inputForAccepted(
            "Agent attitude:",
            lambda: input(f"E.g. \"researches new topics and discusses existing research.\"\n> {row['agent_name']}... "),
            lambda _: print("Accept?")
        ).strip()
        row["agent_attitude"] = user_input
        agent_table.update(row, doc_ids=[1])
    if row.get("user_name") == None:
        user_input = inputForAccepted(
            "Your name:",
            lambda: input("> Name: "),
            lambda _: print("Accept?")
        ).strip()
        row["user_name"] = user_input
        agent_table.update(row, doc_ids=[1])
    print("Agent config is complete")

def configModel():
    row = None
    if len(model_table.all()) > 0:
        row = model_table.all()[0]
    if row == None:
        row = {"active": True}
        model_table.insert(row)
    if row.get("main_model_source") == None:
        user_input = inputForAccepted(
            "Ollama model or GGUF file path for custom main model:",
            lambda: input(f"(Empty for default \'{MAIN_MODEL_DEFAULT}\')> Model or GGUF file path: "),
            lambda _: print("Accept?")
        ).strip()
        if user_input == "":
            user_input = MAIN_MODEL_DEFAULT
        row["main_model_source"] = user_input
        model_table.update(row, doc_ids=[1])
    if row.get("fast_model") == None:
        user_input = inputForAccepted(
            "Small and fast Ollama model for simpler cases:",
            lambda: input(f"(Empty for default \'{FAST_MODEL_DEFAULT}\')> Model: "),
            lambda _: print("Accept?")
        ).strip()
        if user_input == "":
            user_input = FAST_MODEL_DEFAULT
        row["fast_model"] = user_input
        model_table.update(row, doc_ids=[1])
    print("Model config is complete")

def writeModelfile():
    print("Creating Modelfile for Ollama...")
    shutil.copy(MODELFILE_TEMPLATE, MODELFILE_GENERATED)
    agent_table_row = agent_table.all()[0]
    model_table_row = model_table.all()[0]
    mapper = {
        "agent_type": agent_table_row["agent_type"],
        "agent_name": agent_table_row["agent_name"],
        "agent_relation": agent_table_row["agent_relation"],
        "agent_attitude": agent_table_row["agent_attitude"],
        "user_name": agent_table_row["user_name"],
        "main_model_source": model_table_row["main_model_source"]
    }
    print(mapper)
    contents = ""
    with open(MODELFILE_GENERATED, 'r') as file:
        contents = file.read()
        for key, value in mapper.items():
            contents = re.sub(f"\\[{str(key)}\\]", value, contents)
    with open(MODELFILE_GENERATED, 'w') as file:
        file.write(contents)
    

def createModel():
    print("Creating model from modelfile using Ollama...")
    subprocess.check_output(f"ollama create {MAIN_MODEL_NAME} -f {MODELFILE_GENERATED}", shell=True)

def flatten(list_of_dicts):
    result = {}
    for d in list_of_dicts:
        if isinstance(d, dict):
            for k, v in d.items():
                result[k] = v
        elif hasattr(d, '__dict__'):
            for attr_name in dir(d):
                if not attr_name.startswith('__') and not callable(getattr(d, attr_name)):
                    result[attr_name] = getattr(d, attr_name)
    return result

def inputForAccepted(title, generator, confirmation=None):
    isAccepted = False
    data = None
    while not isAccepted:
        print(title)
        data = generator()
        if confirmation != None:
            confirmation(data)
        isAccepted = inputAccepted()
        print()
        if not isAccepted:
            print("-----------------------------")
    return data

def inputAccepted():
    accept = input("> Accept [y/n]?")
    return re.search("y", accept, re.IGNORECASE) != None

if __name__ == "__main__":
    start()
