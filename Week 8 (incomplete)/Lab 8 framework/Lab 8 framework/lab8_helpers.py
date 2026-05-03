# Based on https://github.com/andrewdcampbell/seam-carving
# Explanation of dynamic programming found at https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from numba import jit

SEAM_COLOR = np.array([1, .8, .8])  # seam visualization color (BGR)


#######################################
# UTILITY CODE
#######################################

def visualize(im, boolmask=None, rotate=False):
    if boolmask is not None:
        im[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        im = rotate_image(im, False)
    plt.imshow(im)
    plt.show()


def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (int(h * width / float(w)), width)
    resized = skimage.transform.resize(image, dim)
    return resized


def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)


########################################
# ENERGY FUNCTIONS
########################################

# @jit
def forward_energy(rgb_im, vis, rotate):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = rgb_im.shape[:2]

    im = np.dot(rgb_im, [0.299, 0.587, 0.114])

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    # if vis:
    #     visualize(energy, rotate=rotate)
    #     Image.fromarray((energy * 256).astype(np.uint8)).save("data/forward_energy_demo.jpg")

    return energy


########################################
# SEAM HELPER FUNCTIONS
########################################

# @jit
def add_seam(im, seam_idx):
    """
    Add a vertical seam to a 3-channel color image at the indices provided
    by averaging the pixels values to the left and right of the seam.
    Code adapted from https://github.com/vivianhylee/seam-carving.
    """
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output


# @jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))


########################################
# MAIN ALGORITHM
########################################

def seams_removal(im, num_remove, vis, rot, seam_function):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, vis, rot, seam_function)
        if vis:
            visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
    return im


def seams_insertion(im, num_add, vis, rot, seam_function):
    seams_record = []
    temp_im = im.copy()

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, vis, rot, seam_function)
        if vis:
            visualize(temp_im, boolmask, rotate=rot)

        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        if vis:
            visualize(im, rotate=rot)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2

    return im


########################################
# MAIN DRIVER FUNCTIONS
########################################

def seam_carve(im, dy, dx, visalise_seams, seam_function):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    output = im

    if dx < 0:
        output = seams_removal(output, -dx, visalise_seams, rot=False, seam_function=seam_function)

    elif dx > 0:
        output = seams_insertion(output, dx, visalise_seams, rot=False, seam_function=seam_function)

    if dy < 0:
        output = rotate_image(output, True)
        output = seams_removal(output, -dy, visalise_seams, rot=True, seam_function=seam_function)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        output = seams_insertion(output, dy, visalise_seams, rot=True, seam_function=seam_function)
        output = rotate_image(output, False)

    return output


def calculate_minimum_path_dp(weight_matrix):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    weight_matrix = weight_matrix.copy()

    h, w = weight_matrix.shape[:2]
    backtrack = np.zeros_like(weight_matrix, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(weight_matrix[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = weight_matrix[i - 1, idx + j]
            else:
                idx = np.argmin(weight_matrix[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = weight_matrix[i - 1, idx + j - 1]

            weight_matrix[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    j = np.argmin(weight_matrix[-1])
    for i in range(h - 1, -1, -1):
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return seam_idx


# @jit
def get_minimum_seam(im, vis, rotate, seam_function):
    h, w = im.shape[:2]
    weight_matrix = forward_energy(im, vis, rotate)

    start, end = get_graph(weight_matrix)

    # Ignore start and finish indices
    seam_idx = seam_function(start, end)[1: -1]

    seam_idx = np.array(seam_idx)

    assert len(seam_idx) == h, f"You must return a path of length {h}"

    jumps = np.where(np.abs(seam_idx[1:] - seam_idx[:-1]) > 1)[0]
    assert len(jumps) == 0, f"Your path is discontinuous at column {jumps[0]}"

    boolmask = np.ones((h, w), dtype=np.bool)
    for row, col in enumerate(seam_idx):
        boolmask[row, col] = False

    return seam_idx, boolmask


def get_image(path):
    image = np.array(Image.open(path)).astype(np.float64) / 256
    assert image is not None
    return image


def save_image(output, path):
    Image.fromarray((output * 256).astype(np.uint8)).convert("RGB").save(path)


class Node:
    def __init__(self, data_to_remember):
        self.neighbours = []
        self.visited = False
        self.previous = None
        self.distance = 10000000
        self.data_to_remember = data_to_remember

    def add(self, node, weight):
        self.neighbours.append((node, weight))

    def __lt__(self, other):
        return 1


def get_graph(weight_matrix):
    h, w = weight_matrix.shape

    start = Node(-1)
    start.distance = 0
    end = Node(-2)

    nodes = [[Node(i) for i in range(w)] for j in range(h)]
    for j in range(w):
        start.add(nodes[0][j], weight_matrix[0][j])
        nodes[h - 1][j].add(end, 0)
        for i in range(h - 1):
            if j > 0:
                nodes[i][j].add(nodes[i + 1][j - 1], weight_matrix[i + 1][j - 1])
            nodes[i][j].add(nodes[i + 1][j], weight_matrix[i + 1][j])
            if j < w - 1:
                nodes[i][j].add(nodes[i + 1][j + 1], weight_matrix[i + 1][j + 1])

    return start, end