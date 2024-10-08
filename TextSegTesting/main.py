import cv2
import glob
from pre_processing import preProcessing
import os

# myImage= cv2.imread('pngImgs/t2.png')
# pngImages = glob.glob("pngImgs/*.PNG")
jpgImages = glob.glob("jpgImgs/*.JPG")
# EjpgImages = glob.glob("jepgImgs/*.JPEG")


def main():
    for jpg in jpgImages:
        print(jpg)
        directory = "C:\\Users\\riley\\Documents\\GitHub\\RileyHerbstProject\\SegmentedImages"
        image = cv2.imread(jpg)
        returnImage = preProcessing(image)
        cv2.imshow(f"After Processed Image {jpg}", returnImage)

        # List files and directories
        # in 'C:/Users/Rajnish/Desktop/GeeksforGeeks'
        print("Before saving image:")
        print(os.listdir(directory))

        # Create a folder on the desktop
        desktop_directory = os.path.join(
            os.path.join(os.environ["USERPROFILE"]), "Desktop"
        )
        processed_images_directory = os.path.join(desktop_directory, "Processed_Images")
        os.makedirs(processed_images_directory, exist_ok=True)

        filename = os.path.join(
            processed_images_directory, f"Boxed_{os.path.basename(jpg)}"
        )

        # Using cv2.imwrite() method
        # Saving the image
        cv2.imwrite(filename, returnImage)

        # List files and directories
        # in 'C:/Users / Rajnish / Desktop / GeeksforGeeks'
        print("After saving image:")
        print(os.listdir(directory))

        cv2.waitKey(0)


main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# ------------------------------------------------------------
