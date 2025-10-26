import google.genai
import logging
import inspect

logging.basicConfig(level=logging.INFO)

for name, obj in inspect.getmembers(google.genai):
    if inspect.ismodule(obj):
        logging.info(f"dir({name}): {dir(obj)}")
