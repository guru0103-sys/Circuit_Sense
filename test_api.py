from inference_sdk import InferenceHTTPClient

# Use the credentials from your uploaded image
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="UXUIoT15yWOWw6DXxyYA" 
)

# Replace 'your_image.jpg' with a photo of your circuit
result = CLIENT.infer("your_image.jpg", model_id="electronics-components-cdyjj-wucwm/1")
print(result)
#