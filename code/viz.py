'''
Script for visualizations
'''
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

def visualize_segmented_mask(image, mask):
    '''
    Displays the mask over the image
    '''

    # convert the numpy array to PIL image
    img = Image.fromarray(image).convert('RGBA')
    img.putalpha(200)

    # create the mask overlay
    overlay = Image.new('RGBA', img.size, (255,255,255,0))
    drawing = ImageDraw.Draw(overlay)
    drawing.bitmap((0, 0), Image.fromarray(mask), fill=(100, 0, 255, 75))
    


    output = Image.alpha_composite(img, overlay)

    return output


if __name__ == '__main__':
    input_img = np.asarray(Image.open('data/test/im1.jpg'))

    mask = np.zeros(input_img.shape[:2], dtype=np.uint8)
    mask[1000:2000, 1000:2000] = 255

    print(input_img.shape)
    print(mask.shape)

    output = visualize_segmented_mask(input_img, mask)

    plt.imshow(output)
    plt.show()



