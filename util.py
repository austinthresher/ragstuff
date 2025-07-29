import re
import io
from PIL import Image
from agents import FunctionTool
import base64
import cairosvg

def function_tool_like(
    reference_tool, new_callable, name=None, description=None
) -> FunctionTool:
    """Copies the schema, name, and description from an existing FunctionTool,
    replacing the existing Callable. Optionally overrides the name and/or
    description.
    """
    return FunctionTool(
        name=name or reference_tool.name,
        description=description or reference_tool.description,
        params_json_schema=reference_tool.params_json_schema,
        on_invoke_tool=new_callable,
    )

def extract_xml_tag(tag:str, text:str) -> str:
    pat = f"<{tag}>(.*?)</{tag}>"
    return re.findall(pat, text)

def remove_alpha(image: Image.Image) -> Image.Image:
    bg = Image.new("RGB", image.size, (0, 0, 0))
    image = image.convert("RGBA")
    bg.paste(image, (0, 0), image)
    return bg

def svg_to_pil(svg: str | bytes, size: tuple[int, int] | None = None) -> Image.Image:
    if isinstance(svg, str):
        svg = svg.encode("utf-8")
    output_width, output_height = size if size else (None, None)
    png_data = cairosvg.svg2png(
        svg, output_width=output_width, output_height=output_height
    )
    return Image.open(io.BytesIO(png_data))

def base64_image(image: Image.Image|str) -> str:
    """Accepts either a filename or a PIL Image. Returns it as a base64 encoded"""
    # TODO: Support URLs too?
    if isinstance(image, str):
        image = remove_alpha(Image.open(image))
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_data}"

# Nearly identical, delete the above one once it's no longer used
def encode_image(img):
    img_byte_arr = io.BytesIO()
    remove_alpha(img).save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

# TODO: This should probably use a better metric
def check_image_difference(image1: Image.Image, image2: Image.Image) -> bool:
    im1 = np.array(image1.resize((512, 512)))
    im2 = np.array(image2.resize((512, 512)))
    # Normalize range
    im1 = (im1 - im1.min()) / (im1.max() - im1.min())
    im2 = (im2 - im2.min()) / (im2.max() - im2.min())
    diff = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    mse = diff / float(im1.shape[0] * im1.shape[1])
    print(f"MSE: {mse}")
    return mse > 0.005
