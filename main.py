import cv2
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.



if __name__ == '__main__':
    print_hi('PyCharm')
    im = cv2.imread("cat.jpeg")
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    cv2.imshow("lol",gray )
    cv2.waitKey(0)
    cv2.destroyAllWindows()