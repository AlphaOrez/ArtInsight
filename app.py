# from fastapi import FastAPI, HTTPException
# from PIL import Image
# import requests
# from io import BytesIO
# from fastapi import FastAPI, HTTPException, UploadFile, Form, File
# from PIL import Image
# from io import BytesIO
# from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# from llama_index.multi_modal_llms.generic_utils import (
#     load_image_urls,
# )

# from llama_index.schema import ImageDocument
# from pydantic import ValidationError
# import uvicorn
# import os
# import openai
# from fastapi import FastAPI, HTTPException
# from PIL import Image
# import requests
# from io import BytesIO
# from llama_index.multi_modal_llms.openai import OpenAIMultiModal
# from llama_index.multi_modal_llms.generic_utils import load_image_urls
# from pathlib import Path
# from llama_hub.file.pymu_pdf.base import PyMuPDFReader
# from llama_index import VectorStoreIndex, ServiceContext
# from llama_index.llms import OpenAI

# app = FastAPI()

# def generate_summary(image_url):
#     img_response = requests.get(image_url)
#     img = Image.open(BytesIO(img_response.content))

#     openai_mm_llm = OpenAIMultiModal(
#         model="gpt-4-vision-preview",
#         api_key="sk-",
#         max_new_tokens=3000
#     )

#     complete_response = openai_mm_llm.complete(
#         prompt="""
#         Please provide a comprehensive summary for this image of an art piece:

#         Image Name: Only provide an image name without any description.

#         Image url: Provide the image url of the input image.

#         Description: Begin by describing the visual aspects of the artwork, such as its colors, shapes, composition, and prominent subjects.

#         Artistic Style or Technique: Mention the artistic style or technique used in creating this piece. Is it impressionistic, surrealistic, abstract, or realistic?

#         Interpretation and Meaning: Explore the potential meaning or message conveyed by the artwork. Discuss any symbolism, emotions evoked, or historical/cultural context that might be relevant.

#         Artist Information: Share details about the artist, including their background, artistic approach, and the period in which the artwork was created.

#         Personal Response or Impact: Express your personal thoughts or feelings about the artwork. How does it resonate with you or potentially impact viewers?

#         Additional Context: Provide any additional information about the artwork's significance in art history, where it is housed, or interesting anecdotes related to its creation or reception.

#         Price: Include the estimated price or value of the artwork, if available.

#         Dimensions: Mention the dimensions of the artwork, such as height, width, and depth, to give a sense of its physical size.
#         """,
#         image_documents=load_image_urls([image_url]),
#     )

#     data_dict = {"Text": complete_response.text}
#     sections = [section.strip() for section in data_dict['Text'].split('\n\n') if section]
#     sections_dict = dict(pair.split(': ', 1) for pair in sections)

#     for key, value in sections_dict.items():
#         if isinstance(value, str):
#             sections_dict[key] = value.replace('\"', '')

#     return sections_dict




# def get_chatbot_response(image_file, document_file, user_prompt, prompt_template):
#     try:
#         os.environ["OPENAI_API_KEY"] = "sk-"
#         openai.api_key = os.environ["OPENAI_API_KEY"]
#         # Read image file
#         img = Image.open(BytesIO(image_file.file.read()))

#         # Get the file path of the uploaded document
#         document_file_path = document_file.filename  # Assuming document_file is an UploadFile object

#         # Load PDF document using PyMuPDFReader
#         loader = PyMuPDFReader()
#         documents = loader.load(file_path=document_file_path)

#         # Create OpenAI language model (LLM)
#         gpt35_llm = OpenAI(model="gpt-3.5-turbo")
#         gpt4_llm = OpenAI(model="gpt-4")

#         # Create a VectorStoreIndex from the loaded documents
#         service_context = ServiceContext.from_defaults(chunk_size=1024, llm=gpt35_llm)
#         index = VectorStoreIndex.from_documents(
#             documents, service_context=service_context
#         )

#         # Create a query engine for the index
#         query_engine = index.as_query_engine(similarity_top_k=2)

#         # Generate response using OpenAIMultiModal
#         openai_mm_llm = OpenAIMultiModal(
#             model="gpt-4-vision-preview",
#             api_key="sk-",
#             max_new_tokens=3000
#         )

#         llava_response = openai_mm_llm.complete(
#             prompt=user_prompt,
#             image_documents=load_image_urls([img]),
#         )

#         # Combine OpenAI response with PDF-based query
#         combined_response = llava_response.text + prompt_template
#         rag_response = query_engine.query(combined_response)

#         return {
#             "OpenAI_Response": llava_response.text,
#             "Rag_Response": str(rag_response),
#         }
#     except Exception as e:
#         # Log the exception
#         import logging
#         logging.exception("Error in get_chatbot_response")
#         raise HTTPException(status_code=500, detail="Internal Server Error")


# @app.get("/summary")
# def get_summary(image_url: str):
#     try:
#         summary = generate_summary(image_url)
#         return summary
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

# @app.post("/chatbot")
# async def chatbot(
#     image_file: UploadFile = File(...),
#     document_file: UploadFile = File(...),
#     user_prompt: str = Form(...),
#     prompt_template: str = Form(...),
# ):
#     try:
#         # Get chatbot response using the provided parameters
#         chatbot_response = get_chatbot_response(image_file, document_file, user_prompt, prompt_template)
#         return chatbot_response
#     except Exception as e:
#         # Log the exception
#         import logging
#         logging.exception("Error in chatbot endpoint")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

# # ... (remaining code)

# # Run the FastAPI application

# # ... (remaining code)

# # Run the FastAPI application
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# # def get_chatbot_response(image_url, user_prompt):
# #     # Your existing code for image summarization
# #     # ...

# #     # Chatbot response logic using the user-provided prompt
# #     openai_chatbot_llm = OpenAIMultiModal(
# #         model="gpt-4-vision-preview",
# #         api_key="sk-",  # Replace with your chatbot API key
# #         max_new_tokens=3000
# #     )

# #     complete_response = openai_chatbot_llm.complete(
# #         prompt=user_prompt,
# #         image_documents=load_image_urls([image_url]),
# #         # Add any additional parameters needed for the chatbot
# #     )

# #     data_dict = {"Text": complete_response.text}
# #     # sections = [section.strip() for section in data_dict['Text'].split('\n\n') if section]
# #     # sections_dict = dict(pair.split(': ', 1) for pair in sections)

# #     # for key, value in sections_dict.items():
# #     #     if isinstance(value, str):
# #     #         sections_dict[key] = value.replace('\"', '')

# #     return data_dict
# # def get_chatbot_response(query):
# #     # Simple chatbot response logic using a dictionary
# #     chatbot_responses = {
# #         "hello": "Hello! How can I assist you today?",
# #         "how are you": "I'm just a computer program, but thanks for asking!",
# #         "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
# #         # Add more query-response pairs as needed
# #     }

# #     # Check if the query has a predefined response
# #     response = chatbot_responses.get(query.lower())

# #     if response:
# #         return {"ChatbotResponse": response}
# #     else:
# #         return {"ChatbotResponse": "I'm sorry, I didn't understand that question."}


# # def chatbot(image_url: str, user_prompt: str):
# #     try:
# #         chatbot_response = get_chatbot_response(image_url, user_prompt)
# #         return chatbot_response
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Error getting chatbot response: {str(e)}")



# from fastapi import FastAPI, HTTPException
# from fastapi import FastAPI, HTTPException, Form, File, UploadFile
# from PIL import Image
# import requests
# from io import BytesIO
# from llama_index.multi_modal_llms.openai import OpenAIMultiModal
# from llama_index.multi_modal_llms.generic_utils import load_image_urls
# from pathlib import Path
# from llama_hub.file.pymu_pdf.base import PyMuPDFReader # type: ignore
# from llama_index import VectorStoreIndex, ServiceContext
# from llama_index.llms import OpenAI
# import openai
# import os
# from llama_index.schema import ImageDocument
# import uvicorn
# import requests

# from glob import glob

# app = FastAPI()


# def generate_summary(image_url):
#     img_response = requests.get(image_url)
#     img = Image.open(BytesIO(img_response.content))

#     openai_mm_llm = OpenAIMultiModal(
#         model="gpt-4-vision-preview",
#         api_key="sk-",
#         max_new_tokens=3000
#     )

#     complete_response = openai_mm_llm.complete(
#         prompt="""
#         Please provide a comprehensive summary for this image of an art piece:

#         Image Name: Only provide an image name without any description.

#         Image url: Provide the image url of the input image.

#         Description: Begin by describing the visual aspects of the artwork, such as its colors, shapes, composition, and prominent subjects.

#         Artistic Style or Technique: Mention the artistic style or technique used in creating this piece. Is it impressionistic, surrealistic, abstract, or realistic?

#         Interpretation and Meaning: Explore the potential meaning or message conveyed by the artwork. Discuss any symbolism, emotions evoked, or historical/cultural context that might be relevant.

#         Artist Information: Share details about the artist, including their background, artistic approach, and the period in which the artwork was created.

#         Personal Response or Impact: Express your personal thoughts or feelings about the artwork. How does it resonate with you or potentially impact viewers?

#         Additional Context: Provide any additional information about the artwork's significance in art history, where it is housed, or interesting anecdotes related to its creation or reception.

#         Price: Include the estimated price or value of the artwork, if available.

#         Dimensions: Mention the dimensions of the artwork, such as height, width, and depth, to give a sense of its physical size.
#         """,
#         image_documents=load_image_urls([image_url]),
#     )

#     data_dict = {"Text": complete_response.text}
#     sections = [section.strip() for section in data_dict['Text'].split('\n\n') if section]
#     sections_dict = dict(pair.split(': ', 1) for pair in sections)

#     for key, value in sections_dict.items():
#         if isinstance(value, str):
#             sections_dict[key] = value.replace('\"', '')

#     return sections_dict


# @app.get("/summary")
# def get_summary(image_url: str):
#     try:
#         summary = generate_summary(image_url)
#         return summary
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")



# def download_image(image_url):
#     response = requests.get(image_url)
#     img = Image.open(BytesIO(response.content))
#     return img

# def download_pdf(url, save_folder):
#     response = requests.get(url)
    
#     # Check if the request was successful (status code 200)
#     if response.status_code == 200:
#         # Extract the file name from the URL
#         file_name = url.split("/")[-1]
        
#         # Specify the file path in the "data" folder
#         file_path = os.path.join(save_folder, file_name)
        
#         # Save the PDF file to the specified path
#         with open(file_path, 'wb') as pdf_file:
#             pdf_file.write(response.content)
        
#         return file_path
#     else:
#         # Handle unsuccessful download
#         raise Exception(f"Failed to download PDF from {url}")

# def get_chatbot_response(user_prompt, prompt_template, document_url):
#     try:
#         data_directory = os.path.abspath(os.path.join(os.getcwd(), "data"))
#         downloaded_file_path = download_pdf(document_url, data_directory)
#         os.environ["OPENAI_API_KEY"] = "sk-"
#         openai.api_key = os.environ["OPENAI_API_KEY"]
#         # max_new_tokens=3000

#         # Download image and document
#         # img = download_image(image_url)
#         # document = download_document(document_url)
#         # img_response = requests.get(imageUrl)
#         # img = Image.open(BytesIO(img_response.content))

#         # imageUrl = requests.get("https://upload.wikimedia.org/wikipedia/commons/6/6f/Mural_del_Gernika.jpg")
#         # image = Image.open(imageUrl).convert("RGB")


#         imageUrl="https://upload.wikimedia.org/wikipedia/commons/6/6f/Mural_del_Gernika.jpg"
#         img_response = requests.get(imageUrl)
#         img = Image.open(BytesIO(img_response.content))
#         # docUrl="https://www.nga.gov/content/dam/ngaweb/Education/learning-resources/an-eye-for-art/AnEyeforArt-PabloPicasso.pdf"
#         # doc_response=requests.get(docUrl)
#         # Load PDF document using PyMuPDFReader
#         loader = PyMuPDFReader()
#         documents = loader.load(file_path=downloaded_file_path)

#         # Create OpenAI language model (LLM)
#         gpt35_llm = OpenAI(model="gpt-3.5-turbo")
#         gpt4_llm = OpenAI(model="gpt-4-vision-preview")

#         # Create a VectorStoreIndex from the loaded documents
#         service_context = ServiceContext.from_defaults(chunk_size=1024, llm=gpt35_llm)
#         index = VectorStoreIndex.from_documents(
#             documents, service_context=service_context
#         )

#         # Create a query engine for the index
#         query_engine = index.as_query_engine(similarity_top_k=2)

#         # Generate response using OpenAIMultiModal
#         openai_mm_llm = OpenAIMultiModal(
#             model="gpt-4-vision-preview",
#             api_key="sk-",
#             max_new_tokens=3000
#         )

#         complete_response = openai_mm_llm.complete(
#             prompt=user_prompt,
#             # image_documents=[ImageDocument(image_path=imageUrl)],
#             image_documents=load_image_urls([imageUrl]),
#         )

#         llava_response = complete_response[0] if isinstance(complete_response, tuple) else complete_response
        

#         # Combine OpenAI response with PDF-based query
#         # combined_response =  llava_response.text + prompt_template
#         combined_response =  prompt_template
#         rag_response = query_engine.query(combined_response)

#         return {
#             "OpenAI_Response": str(llava_response),
#             "Rag_Response": str(rag_response),
#         }
#     except Exception as e:
#         # Log the exception
#         import logging
#         logging.exception("Error in get_chatbot_response")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

# @app.post("/chatbot")
# async def chatbot(
#     # imageUrl: str = Form(...),
#     document_url: str = Form(...),
#     # document_url: UploadFile = File(...),
#     user_prompt: str = Form(...),
#     prompt_template: str = Form(...),
# ):
#     try:
#         # Get chatbot response using the provided parameters
#         chatbot_response = get_chatbot_response(user_prompt, prompt_template, document_url)
#         return chatbot_response
#     except Exception as e:
#         # Log the exception
#         import logging
#         logging.exception("Error in chatbot endpoint")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from PIL import Image
from io import BytesIO
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.generic_utils import load_image_urls
from llama_hub.file.pymu_pdf.base import PyMuPDFReader # type: ignore
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import openai
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from llama_index.schema import ImageDocument
app = FastAPI()

# CORS middleware for handling Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can modify this based on your requirements
    allow_credentials=True,
    allow_methods=["*"],    # Allow all methods
    allow_headers=["*"],    # Allow all headers
)

def download_pdf(upload_file, save_folder):
    file_path = os.path.join(save_folder, upload_file.filename)
    with open(file_path, 'wb') as pdf_file:
        pdf_file.write(upload_file.file.read())
    return file_path


def generate_summary(image_file):
    # img_response = requests.get(image_url)
    # img = Image.open(BytesIO(img_response.content))

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview",
        api_key="sk-",
        max_new_tokens=3000
    )
    img_directory = os.path.abspath(os.path.join(os.getcwd(), "data/Imagefile"))
    downloaded_file_path_image = download_pdf(image_file, img_directory)
    imageUrl = downloaded_file_path_image

    complete_response = openai_mm_llm.complete(
        prompt="""
        Please provide a comprehensive summary for this image of an art piece:

        Image Name: Only provide an image name without any description.

        Description: Begin by describing the visual aspects of the artwork, such as its colors, shapes, composition, and prominent subjects.

        Artistic Style or Technique: Mention the artistic style or technique used in creating this piece. Is it impressionistic, surrealistic, abstract, or realistic?

        Interpretation and Meaning: Explore the potential meaning or message conveyed by the artwork. Discuss any symbolism, emotions evoked, or historical/cultural context that might be relevant.

        Artist Information: Share details about the artist, including their background, artistic approach, and the period in which the artwork was created.

        Personal Response or Impact: Express your personal thoughts or feelings about the artwork. How does it resonate with you or potentially impact viewers?

        Additional Context: Provide any additional information about the artwork's significance in art history, where it is housed, or interesting anecdotes related to its creation or reception.

        Price: Include the estimated price or value of the artwork, if available.

        Dimensions: Mention the dimensions of the artwork, such as height, width, and depth, to give a sense of its physical size.
        """,
        image_documents=[ImageDocument(image_path=imageUrl)],
    )

    data_dict = {"Text": complete_response.text}
    sections = [section.strip() for section in data_dict['Text'].split('\n\n') if section]
    sections_dict = dict(pair.split(': ', 1) for pair in sections)

    for key, value in sections_dict.items():
        if isinstance(value, str):
            sections_dict[key] = value.replace('\"', '')

    return sections_dict


@app.post("/summary")
def get_summary(image_file: UploadFile = File(...)):
    try:
        summary = generate_summary(image_file)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


def get_chatbot_response(image_file, user_prompt, prompt_template, document_file):
    try:
        doc_directory = os.path.abspath(os.path.join(os.getcwd(), "data/Docfile"))
        downloaded_file_path_doc = download_pdf(document_file, doc_directory)


        img_directory = os.path.abspath(os.path.join(os.getcwd(), "data/Imagefile"))
        downloaded_file_path_image = download_pdf(image_file, img_directory)

        os.environ["OPENAI_API_KEY"] = "sk-"
        openai.api_key = os.environ["OPENAI_API_KEY"]

        imageUrl = downloaded_file_path_image
        # Load PDF document using PyMuPDFReader
        loader = PyMuPDFReader()
        documents = loader.load(file_path=downloaded_file_path_doc)

        # Create OpenAI language model (LLM)
        gpt35_llm = OpenAI(model="gpt-3.5-turbo")
        gpt4_llm = OpenAI(model="gpt-4-vision-preview")

        # Create a VectorStoreIndex from the loaded documents
        service_context = ServiceContext.from_defaults(chunk_size=1024, llm=gpt35_llm)
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

        # Create a query engine for the index
        query_engine = index.as_query_engine(similarity_top_k=2)

        # Generate response using OpenAIMultiModal
        openai_mm_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview",
            api_key="sk-
            max_new_tokens=3000
        )

        complete_response = openai_mm_llm.complete(
            prompt=user_prompt,
            image_documents=[ImageDocument(image_path=imageUrl)],
        )

        llava_response = complete_response[0] if isinstance(complete_response, tuple) else complete_response

        # Combine OpenAI response with PDF-based query
        combined_response = prompt_template
        rag_response = query_engine.query(combined_response)

        return {
            "OpenAI_Response": str(llava_response),
            "Rag_Response": str(rag_response),
        }
    except Exception as e:
        # Log the exception
        import logging
        logging.exception("Error in get_chatbot_response")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/chatbot")
async def chatbot(
    image_file: UploadFile = File(...),
    document_file: UploadFile = File(...),
    user_prompt: str = Form(...),
    prompt_template: str = Form(...),
):
    try:
        # Get chatbot response using the provided parameters
        chatbot_response = get_chatbot_response(image_file, user_prompt, prompt_template, document_file)
        return chatbot_response
    except Exception as e:
        # Log the exception
        import logging
        logging.exception("Error in chatbot endpoint")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
