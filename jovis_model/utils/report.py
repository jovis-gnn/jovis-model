import os
import glob
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


class ReportMaker:
    def __init__(self, data_dict: dict, image_path: str, max_len: int):
        self.image_path_dict = {
            os.path.basename(p).split(".")[0]: p
            for p in glob.glob(os.path.join(image_path, "*"))
        }
        self.data_dict = data_dict
        self.max_len = max_len

    def get_image(self, product_id: str):
        img_path = self.image_path_dict.get(product_id, None)
        if img_path is None:
            img = Image.fromarray(np.ones((150, 150, 3), dtype=np.uint8) * 255)
            draw = ImageDraw.Draw(img)
            draw.text(xy=(75, 75), text="NO IMAGE", fill=(0, 0, 0))
        else:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((150, 150), Image.LANCZOS)
        return img

    def image_base64(self, im):
        with BytesIO() as buffer:
            im.save(buffer, "jpeg")
            return base64.b64encode(buffer.getvalue()).decode()

    def column_formatter(self, x):
        if isinstance(x, Image.Image):
            return f'<img src="data:image/jpeg;base64,{self.image_base64(x)}">'
        else:
            return x

    def get_html(self, table_string):
        result = f"""
        <html>
            <head>
                <title> visualization report </title>
                <style>
                    table {{
                        border-collapse: collapse;
                        border-style: hidden;
                    }}

                    thead {{
                        background: #1640D6;
                        color: #FFFFFF;
                        height: 50px;
                        border: 0;
                    }}

                    thead > tr > th {{
                        border: 2px solid white;
                        padding: 5px;
                    }}

                    tbody > tr > th {{
                        padding: 10px;
                    }}

                    th, td {{
                        text-align: center;
                        border: 1px solid #F6F1EE;
                        padding: 5px;
                    }}
                </style>
            </head>
            <body>
                {table_string}
            </body>
        </html>
        """
        return result

    def get_content_row(self, content: dict, spare_cell: int):
        res = []
        blank_img = Image.fromarray(np.ones((150, 150, 3), dtype=np.uint8) * 255)

        img_contents = [self.get_image(img_id) for img_id in content["image"]]
        img_contents = [blank_img] * (self.max_len - spare_cell) + img_contents
        img_contents = img_contents[: self.max_len]
        img_contents += [blank_img] * (self.max_len - len(img_contents))
        res.append(img_contents)

        text_contents = [
            "<br>".join(c) if isinstance(c, list) else c for c in content["text"]
        ]
        text_contents = [""] * (self.max_len - spare_cell) + text_contents
        text_contents = text_contents[: self.max_len]
        text_contents += [""] * (self.max_len - len(text_contents))
        res.append(text_contents)

        return res

    def make_report(self, save_path: str, save_name: str, col_names=None):
        html_df = []
        for row_id in list(self.data_dict.keys()):
            spare_cell = self.max_len - 1
            html_df.append([row_id] + [""] * (spare_cell))
            contents: dict = self.data_dict[row_id]
            spare_cell -= 1
            for contents_id in list(contents.keys()):
                html_df.append([""] + [contents_id] + [""] * (spare_cell))
                for content in contents[contents_id]:
                    content_rows = self.get_content_row(content, spare_cell=spare_cell)
                    html_df += content_rows

        if col_names is None:
            col_names = [f"col_{i}" for i in range(self.max_len)]

        html_df = pd.DataFrame(html_df, columns=col_names)
        formatter = {c: self.column_formatter for c in col_names}
        table = html_df.to_html(formatters=formatter, escape=False)
        html = self.get_html(table)
        with open(os.path.join(save_path, f"{save_name}.html"), "w") as f:
            f.write(html)
