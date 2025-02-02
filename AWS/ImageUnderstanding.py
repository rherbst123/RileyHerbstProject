import base64
import boto3
import json
import streamlit as st

def process_image(image_file, prompt):
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
    )

    MODEL_ID = "meta.llama3-2-90b-instruct-v1:0"
    
    # Read and encode the uploaded file
    binary_data = image_file.getvalue()
    base_64_encoded_data = base64.b64encode(binary_data)
    base64_string = base_64_encoded_data.decode("utf-8")
    
    # Define your system prompt(s)
    system_list = [{
        "text": prompt
    }]
    
    # Define a "user" message including both the image and a text prompt
    message_list = [{
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "jpg",
                    "source": {"bytes": base64_string},
                }
            },
            {
                "text": prompt
            }
        ],
    }]
    
    # Configure the inference parameters
    inf_params = {"max_new_tokens": 4096, "top_p": 0.1, "top_k": 20, "temperature": 0.3}

    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }
    
    # Invoke the model and extract the response
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())
    
    return model_response["output"]["message"]["content"][0]["text"]

def main():
    st.title("Label Transcription App")
    
    # Add some description
    st.write("Upload an image to transcribe information")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    # Text input for prompt
    prompt = st.text_input("What would you like to know about the image?")
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image")
        
        # Add a button to process the image
        if st.button("Generate Descriptions"):
            with st.spinner("Generate transcription"):
                try:
                    result = process_image(uploaded_file, prompt)
                    st.write("### Generated Description:")
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
