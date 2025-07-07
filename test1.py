
from utils.memory_client import add_context, get_memories


add_context([{"role":"user","content":"Hello again"}], metadata={"source":"moduleX"})

get_memories("Hello","moti")
