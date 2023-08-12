# %%
import sys
from PIL import Image
from rembg import remove


def remove_by_rembg(filepath: str, max_size: float = 500):
    img = Image.open(filepath)
    img.resize([max_size*i//max(img.width, img.height)
                for i in [img.width, img.height]])
    return remove(img)


if __name__ == '__main__':
    if 'ipykernel_launcher.py' in sys.argv[0]:
        filepath = './data/test.jpg'
    else:
        filepath = sys.argv[1]
    remove_by_rembg(filepath, 300).show()

# %%
