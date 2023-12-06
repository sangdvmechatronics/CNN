import argparse
import cv2
def get_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=int, help = "nhap dau vao", default= 0)
    parser.add_argument("--string_print", type = str, help = "print", default= "True")
    args = parser.parse_args()
    return args

def main():
    args = get_input()
    args_device = args.device
    args_string_print = args.string_print
    cap = cv2.VideoCapture(args_device)
    while True:
        ret, frame = cap.read()
        cv2.imshow("video", frame)
        print(f"--{args_string_print}--")
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()

