import os
import urllib.request

# YOLOv5 weight URLs
weights = {
    "s": {
        "name": "yolov5s.pt",
        "url": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt",
        "desc": "YOLOv5 Small – fastest model, lower accuracy, good for quick experiments."
    },
    "l": {
        "name": "yolov5l.pt",
        "url": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt",
        "desc": "YOLOv5 Large – slower but higher accuracy, good for better detection results."
    }
}

print("\nYOLOv5 Weight Downloader\n")
print("Available models:\n")

for k, v in weights.items():
    print(f"{k.upper()} : {v['name']}")
    print(f"    {v['desc']}\n")

print("Options:")
print("1 -> Download YOLOv5s")
print("2 -> Download YOLOv5l")
print("3 -> Download BOTH\n")

choice = input("Enter your choice (1/2/3): ").strip()

os.makedirs("pretrained", exist_ok=True)

def download(weight):
    path = os.path.join("pretrained", weight["name"])
    print(f"\nDownloading {weight['name']}...")
    urllib.request.urlretrieve(weight["url"], path)
    print(f"Saved to {path}")

if choice == "1":
    download(weights["s"])

elif choice == "2":
    download(weights["l"])

elif choice == "3":
    download(weights["s"])
    download(weights["l"])

else:
    print("Invalid choice. Please run again.")