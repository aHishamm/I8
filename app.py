from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from langchain_community.tools import DuckDuckGoSearchRun
from ai71 import AI71
import gradio as gr
import openai 
import os
import re
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import pytesseract
# Make sure to import the necessary OpenAI API client and configure it.
all_cals = {}
def extract_calories_and_items(text):
    # Use regular expression to find all numerical values associated with "calory" or "calories"
    pattern = r'(\d+)\s*(?:calory|calories)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Convert the matches to integers
    calories = [int(match) for match in matches]
    
    return calories

def plot_calories(calories):
    labels = sorted(calories, key=calories.get)
    vals = [calories[label] for label in labels]
    plt.barh(labels, vals, color='skyblue')
    plt.xlabel('Calories')
    plt.title('Item and Count')
    plt.tight_layout()
    
def parse_items(items_string):
    # Remove square brackets and split by comma
    items_list = items_string.strip('[]').split(',')
    
    item_dict = {}
    
    # Define the pattern to match the quantity and item
    pattern = r'(\d+)\s*x\s*(\w+)'
    
    for item in items_list:
        match = re.match(pattern, item.strip())
        if match:
            quantity = int(match.group(1))
            item_name = match.group(2)
            if item_name in item_dict:
                item_dict[item_name] += quantity
            else:
                item_dict[item_name] = quantity
    
    return item_dict

# Set the API key for AI71
#AI71_API_KEY = "key"
AI71_API_KEY = os.getenv('KEY')
AI71_BASE_URL = "https://api.ai71.ai/v1/"
client = AI71(AI71_API_KEY)

search = DuckDuckGoSearchRun()

# usr_input = input(f"User:")
# 
# print(items)



def chatGPT_food(userinput, temperature=0.1, max_tokens=300):

    keyword = client.chat.completions.create(
        model="tiiuae/falcon-180B-chat",
        messages=[
            {"role": "system", "content": '''you need to extract the food item from the user text without any comments
             example: 
             user: I ate two apples
             assistant: 2 x apple'''},
            {"role": "user", "content": userinput}
        ],
        # temperature=0.5, 
    )

    items = parse_items(keyword.choices[0].message.content)
    
    for item, count in items.items():
        result = search.invoke(f'calories of {item}')

        response = client.chat.completions.create(
                model="tiiuae/falcon-180B-chat",
                messages=[
                    {"role": "system", "content": '''based on the provided information extract the calories count per portion of the item provided, just the calories and portion in grams or ml without further comments 
                        Example:
                        orange 47 calories per 100 gram
                        cola 38 calories per 100 gram
                        do not generate more or add any unneeded comments, just follow the examples strictly'''},
                    {"role": "user", "content": result}
                ],
                temperature=0.2, 
            )

    # print("search")
    # print(result)
    # print("ai")
    # print (response.choices[0].message.content)
        calories = extract_calories_and_items(response.choices[0].message.content)
        # print("calories")
        # print(calories)
        try:
            all_cals[f"{count}x{item}"] = count*calories[0]
        except:
            continue
        return all_cals

def chatGPT_invoice(userinput, temperature=0.1, max_tokens=300): 
    response = client.chat.completions.create(
        model="tiiuae/falcon-180B-chat",
        messages=[
            {"role": "system", "content": '''from the following invoice, find the name of the restaurant, then write a table for each food in the invoice and estimate its calories count only knowing that this food is from the same restaurant, with no further text or comments, or notes:
            example:
            "Restaurant: KFC
            <insert the table of food and estimated calories>"
            Do it for this text:'''},
            {"role": "user", "content": userinput}
        ],
        temperature=temperature, 
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
  
def update_plot(userinput):
    # all_cals = chatGPT_food(userinput)
    fig, ax = plt.subplots()
    plot_calories(all_cals)
    return fig

def ocr(input_img):
    img1 = np.array(input_img)
    text = pytesseract.image_to_string(img1)
    output = chatGPT_invoice(text)
    return output

with gr.Blocks() as demo:
    with gr.Tab("Food Calories"):
        food = gr.Textbox(label="Food")
        output = gr.Textbox(label="Calories")
        greet_btn = gr.Button("Get Calories")
        greet_btn.click(fn=chatGPT_food, inputs=food, outputs=output)

    with gr.Tab("Invoice OCR"):
        image_input = gr.Image(height=200, width=200)
        output_text = gr.Textbox(label="Estimated Calories from Invoice")
        demo_ocr = gr.Interface(fn=ocr, inputs=image_input, outputs=output_text)

    with gr.Tab("Calories Plot"):
        # food_plot = gr.Textbox(label="Enter Food for Plot")
        plot_output = gr.Plot(label="Calories Plot")
        plot_btn = gr.Button("Generate Plot")
        plot_btn.click(fn=update_plot, inputs=plot_btn, outputs=plot_output)

demo.launch()
