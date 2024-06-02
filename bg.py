import base64

with open("background.png", "rb") as image_file:
    base64_str = base64.b64encode(image_file.read()).decode()

print(base64_str)
