import cv2
import os

def save_pred_(cap, img, args):
    if args.save_mode =="image":
        img_path = os.path.join(args.save_path, "images.jpg")
        cv2.imwrite(img_path, img)
    elif args.save_mode == "video":

        if not os.path.exists(os.path.join(args.save_path, "video_1.mp4")):
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(os.path.join(args.save_path, "video_1.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        else:
            out.write(img)

    else:
        print("Warning camera capture")
