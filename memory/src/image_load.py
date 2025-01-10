import os

path = "/Users/kuzumochi/go2/nav/memory/src/images/test"

image_files = os.listdir(path)
images = [os.path.join(path, file_name) for file_name in image_files]

print(images)