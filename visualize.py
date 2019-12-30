# visualize bounding box
import cv2


def test():
    img_path = "C:\\Users\\liuhui\\Desktop\\set00_V000_68.jpg"
    bbox = [[213, 141, 213 + 5, 141 + 9]]
    img = cv2.imread(img_path)
    print(img.shape)
    for b in bbox:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
    cv2.imwrite('test.jpg', img)
    cv2.imshow('test', img)


if __name__ == '__main__':
    test()
