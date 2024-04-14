import numpy as np
import cv2 as cv


class Point:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

    def get_data(self):
        return [self.data[0], self.data[1]]

    def copy(self):
        new_p = Point(self.data)
        new_p.prev = self.prev
        new_p.next = self.next

        return new_p


class DoublyLinkedPoints:
    def __init__(self):
        self.head = None
        self.last = None
        self.data_counter = 0
        self.points_as_array = []

    def insert(self, p):
        """

        :param p: touple contains data
        :return:
        """
        new_p = Point(p)

        if self.head == None:
            self.head = new_p
            self.last = new_p
        else:
            self.last.next = new_p
            new_p.prev = self.last
            self.last = self.last.next

        self.data_counter += 1

    def pop(self):
        """
        delete last point and return it
        :return:
        """
        if self.last is None:
            return None

        popped = self.last.copy()
        self.last = self.last.prev
        if self.last is None:
            self.head = None
        else:
            self.last.next = None

        self.data_counter -= 1
        return popped

    def len(self):
        return self.data_counter

    def get_points_as_array(self):
        p = self.head
        arr = []
        while p is not None:
            arr.append(p.get_data())
            p = p.next
        return np.array(arr)

    def copy(self):
        new_dp = DoublyLinkedPoints()
        p = self.head

        while p is not None:
            new_dp.insert(p.data)
            p = p.next

        return new_dp

    def update_cell_val(self, val, index):
        """
        update the points cordinates at index in the list
        :param val:
        :param index:
        :return:
        """
        tmp = self.head
        pos = 0
        while tmp is not None:
            if pos == index:
                tmp.data = val
                break
            tmp = tmp.next
            pos += 1

    def pass_through(self):
        if self.head == None:
            return
        tmp = self.head
        while tmp is not None:
            print(tmp.data)
            tmp = tmp.next

    def back_through(self):
        if self.last == None:
            return

        tmp = self.last
        while tmp is not None:
            print(tmp.data)
            tmp = tmp.prev


def calculate_distance(img, dist=cv.DIST_L2, kernel_size=5):
    """
    calculate the distance transform of the given img

    :return:
    """

    # img = cv.imread(img_path)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image
    # _, threshold = cv.threshold(grey, 123, 255, cv.THRESH_BINARY)
    # cv.imshow("thres", threshold)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    distTransform = cv.distanceTransform(grey, dist, kernel_size)
    # distTransform.astype('uint8')
    return distTransform

def show_destroy(img):
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#
# mask_path = 'testing/bowl1_annotated/999579001.jpg'
# img = cv.imread(mask_path)
# print(f'img: {img.shape}')
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     #357159
#
# print(img.shape)
#
# distTransform= cv.distanceTransform(img, cv.DIST_L2, 5)
# cv.imshow("img", img)
# cv.imshow("distance", distTransform)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
